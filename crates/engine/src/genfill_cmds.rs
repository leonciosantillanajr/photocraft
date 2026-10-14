//! Generative Fill / Remove / Fill Empty Areas: the bridge between a document and
//! `photocraft-genfill`.
//!
//! The work is split so a UI can keep the window responsive while the network call runs:
//! [`prepare`] reads the document (fast), [`Job::run`] talks to the API (slow, needs no session, run
//! it on a worker thread) and [`apply`] adds the result as new layers in ONE undo step. The
//! `edit.generative*` commands simply run the three in a row.

use std::sync::{Arc, Mutex, PoisonError};

use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerId};
use photocraft_genfill::{Backend, Config, Gray, Mode, Output, Params, Rgba8};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// Largest canvas (pixels) the feature reads into memory.
const MAX_PIXELS: u64 = 150_000_000;
const MAX_REFERENCES: usize = 8;
const MAX_REFERENCE_BYTES: u64 = 64 * 1024 * 1024;

static API_KEY: Mutex<String> = Mutex::new(String::new());
static BACKEND: Mutex<Option<Arc<dyn Backend>>> = Mutex::new(None);

/// The API key from Preferences / the UI; `OPENAI_API_KEY` in the environment takes precedence.
pub fn set_api_key(key: &str) {
    *API_KEY.lock().unwrap_or_else(PoisonError::into_inner) = key.trim().to_string();
}

/// Replaces the network backend (tests use a fake; `None` goes back to OpenAI).
pub fn set_backend_override(backend: Option<Arc<dyn Backend>>) {
    *BACKEND.lock().unwrap_or_else(PoisonError::into_inner) = backend;
}

/// The backend to call: the override, else OpenAI with the key from the environment or [`set_api_key`].
pub fn backend() -> Result<Arc<dyn Backend>> {
    if let Some(b) = BACKEND.lock().unwrap_or_else(PoisonError::into_inner).clone() {
        return Ok(b);
    }
    let key = API_KEY.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let real = photocraft_genfill::openai::UreqBackend::from_env_or(&key).map_err(into_engine)?;
    Ok(Arc::new(real))
}

fn into_engine(e: photocraft_genfill::GenFillError) -> EngineError {
    EngineError::Other(e.0)
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Fill,
    Remove,
    FillEmpty,
}

impl Kind {
    pub fn command(self) -> &'static str {
        match self {
            Kind::Fill => "edit.generativeFill",
            Kind::Remove => "edit.generativeRemove",
            Kind::FillEmpty => "edit.generativeFillEmpty",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Kind::Fill => "Generative Fill",
            Kind::Remove => "Generative Remove",
            Kind::FillEmpty => "Fill Empty Areas",
        }
    }
}

/// Everything a generation needs, copied out of the document.
pub struct Job {
    pub kind: Kind,
    image: Rgba8,
    selection: Option<Gray>,
    text: String,
    references: Vec<Rgba8>,
    tier: Option<String>,
    max_regions: usize,
}

/// The merged (flattened) image as straight RGBA8; transparent where the document is.
fn flatten_rgba8(doc: &Document) -> Result<Rgba8> {
    let r = doc.bounds();
    let (w, h) = (r.width(), r.height());
    if w == 0 || h == 0 || u64::from(w) * u64::from(h) > MAX_PIXELS {
        return Err(EngineError::Other(format!("Imagem grande demais para o Preenchimento Generativo ({w}×{h} px).")));
    }
    let fmt = doc.pixel_format();
    let surface = photocraft_compose::flatten_to_surface(doc, PixelFormat::new(fmt.mode, fmt.sample, true), None);
    let mut buf = vec![[0u8; 4]; w as usize * h as usize];
    surface.read_rgba8_into(r, &mut buf);
    Rgba8::new(w, h, buf.into_iter().flatten().collect()).map_err(into_engine)
}

fn selection_mask(doc: &Document) -> Result<Gray> {
    let sel = doc.selection.as_ref().ok_or_else(|| EngineError::Other("Não há nada selecionado. Selecione a área primeiro.".into()))?;
    let r = doc.bounds();
    let mut buf = vec![[0u8; 4]; r.width() as usize * r.height() as usize];
    sel.read_rgba8_into(r, &mut buf);
    Gray::new(r.width(), r.height(), buf.into_iter().map(|p| p[0]).collect()).map_err(into_engine)
}

fn load_references(cmd: &str, p: &Value) -> Result<Vec<Rgba8>> {
    let list = match p.get("references") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) => a,
        Some(_) => return Err(bad(cmd, "`references` must be a list of image file paths")),
    };
    if list.len() > MAX_REFERENCES {
        return Err(bad(cmd, format!("at most {MAX_REFERENCES} reference images")));
    }
    let mut out = Vec::with_capacity(list.len());
    for v in list {
        let path = v.as_str().ok_or_else(|| bad(cmd, "`references` must be a list of image file paths"))?;
        let meta = std::fs::metadata(path).map_err(|e| EngineError::Other(format!("Não consegui abrir a referência: {path}\n{e}")))?;
        if meta.len() > MAX_REFERENCE_BYTES {
            return Err(EngineError::Other(format!("A referência é grande demais: {path}")));
        }
        let bytes = std::fs::read(path).map_err(|e| EngineError::Other(format!("Não consegui abrir a referência: {path}\n{e}")))?;
        let name = std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or("reference.png");
        let imported = photocraft_io::import(name, &bytes).map_err(|e| EngineError::Other(format!("Não consegui abrir a referência: {path}\n{e}")))?;
        out.push(flatten_rgba8(&imported.document)?);
    }
    Ok(out)
}

fn tier_param(cmd: &str, p: &Value) -> Result<Option<String>> {
    match p.get("quality") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_str().map(str::to_lowercase).as_deref() {
            Some("auto" | "") => Ok(None),
            Some("baixa" | "low") => Ok(Some("baixa".into())),
            Some("media" | "média" | "medium") => Ok(Some("media".into())),
            Some("alta" | "high") => Ok(Some("alta".into())),
            _ => Err(bad(cmd, "`quality` must be auto, baixa, media or alta")),
        },
    }
}

/// Reads the active document and `p` into a [`Job`] (no network).
pub fn prepare(s: &Session, kind: Kind, p: &Value) -> Result<Job> {
    let cmd = kind.command();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let text = match p.get("prompt") {
        None | Some(Value::Null) => String::new(),
        Some(v) => v.as_str().ok_or_else(|| bad(cmd, "`prompt` must be text"))?.to_string(),
    };
    if text.chars().count() > 4000 {
        return Err(bad(cmd, "`prompt` is too long (4000 characters at most)"));
    }
    let max_regions = match p.get("maxRegions") {
        None | Some(Value::Null) => 8,
        Some(v) => v.as_u64().filter(|n| (1..=16).contains(n)).ok_or_else(|| bad(cmd, "`maxRegions` must be a whole number from 1 to 16"))? as usize,
    };
    let selection = if kind == Kind::FillEmpty { None } else { Some(selection_mask(&d.doc)?) };
    Ok(Job {
        kind,
        image: flatten_rgba8(&d.doc)?,
        selection,
        text: if kind == Kind::Remove { String::new() } else { text },
        references: if kind == Kind::Remove { Vec::new() } else { load_references(cmd, p)? },
        tier: tier_param(cmd, p)?,
        max_regions,
    })
}

impl Job {
    /// Calls the API. Touches no session, so it can run on a worker thread.
    pub fn run(&self, backend: &dyn Backend, progress: &dyn Fn(&str)) -> Result<Vec<Output>> {
        let cfg = Config::default();
        let params = Params {
            mode: if self.kind == Kind::Remove { Mode::Remove } else { Mode::Fill },
            text: &self.text,
            references: &self.references,
            tier: self.tier.as_deref(),
        };
        match (&self.selection, self.kind) {
            (Some(sel), Kind::Fill | Kind::Remove) => {
                photocraft_genfill::generate(&self.image, sel, &params, &cfg, backend, progress).map(|o| vec![o]).map_err(into_engine)
            }
            _ => photocraft_genfill::generate_empty(&self.image, &params, &cfg, backend, self.max_regions, progress).map_err(into_engine),
        }
    }
}

/// Adds each result as a new pixel layer above the active one, all in a single undo step.
/// Returns the new layer ids.
pub fn apply(s: &mut Session, kind: Kind, outs: &[Output]) -> Result<Vec<LayerId>> {
    if outs.is_empty() {
        return Err(EngineError::Other("Nada foi gerado.".into()));
    }
    let ids = s.edit(kind.label(), |doc, active| {
        let mut ids = Vec::with_capacity(outs.len());
        for o in outs {
            let mut layer = Layer::raster(doc.next_layer_name("Generative Fill"), doc.pixel_format());
            let surface = layer.surface_mut().ok_or_else(|| EngineError::Other("a new pixel layer has no pixels".into()))?;
            let fmt = surface.format();
            let n = surface.channels();
            let (w, h) = (o.patch.w, o.patch.h);
            let mut data = Vec::with_capacity(w as usize * h as usize * n);
            let mut tmp = [0f32; 8];
            for px in o.patch.px.as_chunks::<4>().0 {
                let rgba = [px[0], px[1], px[2], px[3]].map(|v| f32::from(v) / 255.0);
                let k = photocraft_raster::from_rgba_into(&fmt, rgba, &mut tmp);
                if k != n {
                    return Err(EngineError::Other("unexpected pixel layout in the new layer".into()));
                }
                data.extend_from_slice(&tmp[..k]);
            }
            let (x, y) = (i32::try_from(o.origin.0).unwrap_or(0), i32::try_from(o.origin.1).unwrap_or(0));
            let rect = Rect::new(x, y, x.saturating_add(w as i32), y.saturating_add(h as i32));
            surface.write_region(rect, &data);
            let id = doc.insert_above(*active, layer);
            crate::clip_group::settle(doc, &[id], &Default::default(), false);
            *active = Some(id);
            ids.push(id);
        }
        Ok(ids)
    })?;
    for id in &ids {
        crate::layer_multi_cmds::note_inert_insert(s, *id);
    }
    Ok(ids)
}

fn run_kind(s: &mut Session, kind: Kind, p: &Value) -> Result<Value> {
    let job = prepare(s, kind, p)?;
    let backend = backend()?;
    let outs = job.run(&*backend, &|_| {})?;
    let layers = apply(s, kind, &outs)?;
    let warning = outs.iter().map(|o| o.warning.as_str()).find(|w| !w.is_empty()).unwrap_or("");
    Ok(json!({
        "layers": layers.iter().map(|id| id.0).collect::<Vec<_>>(),
        "areas": outs.len(),
        "plan": outs.first().map(|o| o.plan.describe()),
        "warning": warning,
    }))
}

fn has_selection(s: &Session) -> std::result::Result<(), String> {
    s.active().filter(|d| d.doc.selection.is_some()).map(|_| ()).ok_or_else(|| "no selection".into())
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "edit.generativeFill",
            label: "Generative Fill…",
            menu: &[],
            shortcut: None,
            params: r##"{"prompt":text,"references":[path],"quality":auto|baixa|media|alta} (generates into the selection as a new layer; needs OPENAI_API_KEY; desktop only)"##,
            enabled: has_selection,
            run: |s, p| run_kind(s, Kind::Fill, p),
            journal: false,
        },
        CommandSpec {
            id: "edit.generativeRemove",
            label: "Generative Remove",
            menu: &[],
            shortcut: None,
            params: r##"{"quality":auto|baixa|media|alta} (removes what is in the selection and rebuilds the background, as a new layer; needs OPENAI_API_KEY; desktop only)"##,
            enabled: has_selection,
            run: |s, p| run_kind(s, Kind::Remove, p),
            journal: false,
        },
        CommandSpec {
            id: "edit.generativeFillEmpty",
            label: "Fill Empty Areas",
            menu: &[],
            shortcut: None,
            params: r##"{"prompt":text,"references":[path],"quality":auto|baixa|media|alta,"maxRegions":int=8} (fills transparent areas of the merged image, one new layer per area; needs OPENAI_API_KEY; desktop only)"##,
            enabled: has_doc,
            run: |s, p| run_kind(s, Kind::FillEmpty, p),
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use photocraft_genfill::{BackendError, EditRequest};

    use super::*;

    /// Tests share the process-wide backend override, so they run one at a time.
    static SERIAL: StdMutex<()> = StdMutex::new(());

    struct Fake {
        calls: StdMutex<Vec<EditRequest>>,
        fail: bool,
    }

    impl Backend for Fake {
        fn edit(&self, req: &EditRequest) -> std::result::Result<Vec<u8>, BackendError> {
            self.calls.lock().unwrap().push(req.clone());
            if self.fail {
                return Err(BackendError::Other("sem internet".into()));
            }
            let (w, h) = req.size;
            let px: Vec<u8> = [250u8, 10, 10].iter().copied().cycle().take((w * h * 3) as usize).collect();
            Ok(photocraft_genfill::imgops::encode_png(w, h, png::ColorType::Rgb, &px).unwrap())
        }
    }

    fn install(fail: bool) -> (std::sync::MutexGuard<'static, ()>, Arc<Fake>) {
        let guard = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let fake = Arc::new(Fake { calls: StdMutex::new(Vec::new()), fail });
        set_backend_override(Some(fake.clone()));
        (guard, fake)
    }

    fn doc_session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 600, "height": 500})).unwrap();
        s
    }

    fn layer_count(s: &Session) -> usize {
        s.active().map(|d| d.doc.layers.len()).unwrap_or(0)
    }

    #[test]
    fn fill_adds_one_layer_with_the_generated_pixels_and_undoes_in_one_step() {
        let (_g, fake) = install(false);
        let mut s = doc_session();
        s.execute("select.rect", json!({"x": 200, "y": 150, "width": 120, "height": 100})).unwrap();
        let before = layer_count(&s);
        let r = s.execute("edit.generativeFill", json!({"prompt": "um sol"})).unwrap();
        assert_eq!(layer_count(&s), before + 1);
        assert_eq!(r["areas"], 1);
        assert!(fake.calls.lock().unwrap()[0].prompt.contains("create: um sol"));
        // the new layer is the active one and is red in the middle of the selection
        let d = s.active().unwrap();
        let id = d.active_layer.unwrap();
        let px = d.doc.layer(id).unwrap().surface().unwrap().rgba(260, 200);
        assert!(px[0] > 0.9 && px[1] < 0.1 && px[3] > 0.99, "{px:?}");
        assert!(d.doc.layer(id).unwrap().surface().unwrap().rgba(5, 5)[3] < 0.01);
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(layer_count(&s), before);
        set_backend_override(None);
    }

    #[test]
    fn remove_ignores_the_prompt() {
        let (_g, fake) = install(false);
        let mut s = doc_session();
        s.execute("select.rect", json!({"x": 100, "y": 100, "width": 80, "height": 80})).unwrap();
        s.execute("edit.generativeRemove", json!({"prompt": "ignorado"})).unwrap();
        assert!(fake.calls.lock().unwrap()[0].prompt.starts_with("Remove whatever"));
        set_backend_override(None);
    }

    #[test]
    fn fill_needs_a_selection_and_fill_empty_needs_empty_areas() {
        let (_g, fake) = install(false);
        let mut s = doc_session();
        assert!(s.execute("edit.generativeFill", json!({})).is_err());
        let e = s.execute("edit.generativeFillEmpty", json!({})).unwrap_err().to_string();
        assert!(e.contains("áreas vazias"), "{e}");
        assert!(fake.calls.lock().unwrap().is_empty());
        set_backend_override(None);
    }

    #[test]
    fn fill_empty_fills_transparent_areas_one_layer_each() {
        let (_g, fake) = install(false);
        let mut s = doc_session();
        // hide the Background so the canvas is transparent, then paint two islands on a new layer
        s.edit("holes", |doc, active| {
            let id = active.unwrap();
            let surf = doc.layer_mut(id).unwrap().surface_mut().unwrap();
            let fmt = surf.format();
            let clear = photocraft_raster::from_rgba(&fmt, [0.0, 0.0, 0.0, 0.0]);
            surf.fill_rect(Rect::new(0, 0, 60, 60), &clear);
            surf.fill_rect(Rect::new(400, 300, 520, 420), &clear);
            Ok(())
        })
        .unwrap();
        let before = layer_count(&s);
        let r = s.execute("edit.generativeFillEmpty", json!({})).unwrap();
        assert_eq!(r["areas"], 2);
        assert_eq!(layer_count(&s), before + 2);
        assert_eq!(fake.calls.lock().unwrap().len(), 2);
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(layer_count(&s), before);
        set_backend_override(None);
    }

    #[test]
    fn backend_failure_leaves_the_document_untouched() {
        let (_g, _fake) = install(true);
        let mut s = doc_session();
        s.execute("select.rect", json!({"x": 100, "y": 100, "width": 80, "height": 80})).unwrap();
        let before = layer_count(&s);
        let e = s.execute("edit.generativeFill", json!({})).unwrap_err().to_string();
        assert!(e.contains("sem internet"), "{e}");
        assert_eq!(layer_count(&s), before);
        set_backend_override(None);
    }

    #[test]
    fn bad_params_are_errors_not_panics() {
        let (_g, fake) = install(false);
        let mut s = doc_session();
        s.execute("select.rect", json!({"x": 100, "y": 100, "width": 80, "height": 80})).unwrap();
        for p in [
            json!({"prompt": 5}),
            json!({"prompt": "x".repeat(5000)}),
            json!({"quality": "ultra"}),
            json!({"quality": 3}),
            json!({"references": "a.png"}),
            json!({"references": [1]}),
            json!({"references": ["/definitely/not/here.png"]}),
            json!({"maxRegions": 0}),
            json!({"maxRegions": "x"}),
        ] {
            assert!(s.execute("edit.generativeFill", p.clone()).is_err(), "{p}");
        }
        assert!(fake.calls.lock().unwrap().is_empty());
        set_backend_override(None);
    }

    #[test]
    fn without_a_key_the_message_says_what_to_do() {
        let _g = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        set_backend_override(None);
        if std::env::var("OPENAI_API_KEY").is_ok_and(|k| !k.trim().is_empty()) {
            return; // a real key is configured on this machine
        }
        set_api_key("");
        let mut s = doc_session();
        s.execute("select.rect", json!({"x": 100, "y": 100, "width": 80, "height": 80})).unwrap();
        let e = s.execute("edit.generativeFill", json!({})).unwrap_err().to_string();
        assert!(e.contains("OPENAI_API_KEY"), "{e}");
    }
}
