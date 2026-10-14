//! The whole generative-fill pass: plan → request → patch.

use crate::imgops::encode_png;
use crate::openai::{self, Backend};
use crate::plan::{Config, Plan, make_plan};
use crate::prompt::build_prompt;
use crate::{GenFillError, Gray, Result, Rgba8};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Generate what the prompt describes (or continue the scene when the prompt is empty).
    Fill,
    /// Remove what is in the selection and rebuild the background.
    Remove,
}

pub struct Params<'a> {
    pub mode: Mode,
    pub text: &'a str,
    /// Decoded reference images (at most `Config::max_references` are sent).
    pub references: &'a [Rgba8],
    /// Force a quality tier (`baixa`, `media`, `alta`); `None` picks by crop size.
    pub tier: Option<&'a str>,
}

/// The generated pixels: image-space position, RGBA (alpha = feathered selection) and the plan used.
#[derive(Clone, Debug)]
pub struct Output {
    pub origin: (u32, u32),
    pub patch: Rgba8,
    pub plan: Plan,
    pub warning: String,
}

fn luma(p: &[u8]) -> f32 {
    0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}

/// Warns when the model barely changed the selected area.
fn unchanged_warning(generated: &Rgba8, crop: &Rgba8, ctx_mask: &Gray) -> String {
    let (Ok(g), Ok(c), Ok(m)) = (generated.resize(256, 256), crop.resize(256, 256), ctx_mask.resize_nearest(256, 256)) else {
        return String::new();
    };
    let (mut sum, mut n) = (0.0f32, 0u32);
    for ((gp, cp), &mv) in g.px.as_chunks::<4>().0.iter().zip(c.px.as_chunks::<4>().0.iter()).zip(&m.v) {
        if mv > 127 {
            sum += (luma(gp) - luma(cp)).abs();
            n += 1;
        }
    }
    if n == 0 || sum / n as f32 >= 3.0 {
        return String::new();
    }
    "A IA quase não alterou a área selecionada. Se era uma remoção sobre fundo liso, está tudo certo; senão, tente gerar de novo.".into()
}

fn edit_mask_png(selected: &Gray) -> Result<Vec<u8>> {
    // OpenAI edits the TRANSPARENT part of the mask.
    let px: Vec<u8> = selected.v.iter().flat_map(|&s| [255, 255, 255, if s > 127 { 0 } else { 255 }]).collect();
    encode_png(selected.w, selected.h, png::ColorType::Rgba, &px)
}

fn reference_png(r: &Rgba8, max_side: u32) -> Result<Vec<u8>> {
    let side = r.w.max(r.h);
    if side > max_side && max_side > 0 {
        let s = max_side as f64 / side as f64;
        let (nw, nh) = (((r.w as f64 * s).round() as u32).max(1), ((r.h as f64 * s).round() as u32).max(1));
        r.resize(nw, nh)?.to_png_rgb()
    } else {
        r.to_png_rgb()
    }
}

/// Runs one generative fill / remove for `selected` (same size as `image`, 255 = selected).
pub fn generate(image: &Rgba8, selected: &Gray, params: &Params, cfg: &Config, backend: &dyn Backend, progress: &dyn Fn(&str)) -> Result<Output> {
    if (image.w, image.h) != (selected.w, selected.h) {
        return Err(GenFillError::new("A seleção não tem o mesmo tamanho da imagem."));
    }
    let plan = make_plan(selected, cfg, params.tier)?;
    progress(&plan.describe());

    let crop = image.crop(plan.ctx_box)?;
    let (sw, sh) = plan.send_size;
    let crop_send = if (sw, sh) == (crop.w, crop.h) { crop.clone() } else { crop.resize(sw, sh)? };
    let grow_send = ((cfg.mask_grow_px as f64 * sw as f64 / crop.w as f64).round() as u32 + 1).max(2);
    let ctx_mask = selected.binarize().crop(plan.ctx_box)?;
    let m_send = ctx_mask.resize_nearest(sw, sh)?.grow(grow_send);

    let refs = params.references.iter().take(cfg.max_references).map(|r| reference_png(r, cfg.reference_max_side_px)).collect::<Result<Vec<_>>>()?;
    let prompt = build_prompt(params.mode, params.text, !refs.is_empty());
    let mut images = vec![crop_send.to_png_rgb()?];
    images.extend(refs);

    progress(&format!("Enviando à OpenAI ({})…", plan.tier.model));
    let bytes = openai::call(backend, cfg, &plan.tier, prompt, images, edit_mask_png(&m_send)?, plan.send_size)?;

    progress("Montando a camada…");
    let made = crate::imgops::decode_png(&bytes)?;
    let (cw, ch) = plan.ctx_size();
    let made = if (made.w, made.h) == (cw, ch) { made } else { made.resize(cw, ch)? };
    let warning = unchanged_warning(&made, &crop, &ctx_mask);

    let feather = (cw.max(ch) as f64 * cfg.feather_ratio).max(1.0);
    let grow = cfg.mask_grow_px.max((feather * 1.5).ceil() as u32);
    let soft = ctx_mask.grow(grow).blur(feather as f32);
    let mut patch = made;
    for (p, &a) in patch.px.as_chunks_mut::<4>().0.iter_mut().zip(&soft.v) {
        p[3] = a;
    }
    Ok(Output { origin: (plan.ctx_box[0], plan.ctx_box[1]), patch, plan, warning })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::openai::{BackendError, EditRequest};

    /// Returns a flat colour of the requested size and records the request.
    struct Fake {
        colour: [u8; 3],
        seen: Mutex<Vec<EditRequest>>,
        fail: Option<BackendError>,
    }

    impl Fake {
        fn new(colour: [u8; 3]) -> Self {
            Self { colour, seen: Mutex::new(Vec::new()), fail: None }
        }
    }

    impl Backend for Fake {
        fn edit(&self, req: &EditRequest) -> std::result::Result<Vec<u8>, BackendError> {
            self.seen.lock().unwrap().push(req.clone());
            if let Some(e) = &self.fail {
                return Err(e.clone());
            }
            let (w, h) = req.size;
            let px: Vec<u8> = self.colour.iter().copied().cycle().take((w * h * 3) as usize).collect();
            Ok(encode_png(w, h, png::ColorType::Rgb, &px).unwrap())
        }
    }

    fn photo(w: u32, h: u32) -> Rgba8 {
        let px = (0..w * h).flat_map(|i| [(i % 200) as u8, 80, 120, 255]).collect();
        Rgba8::new(w, h, px).unwrap()
    }

    fn sel(w: u32, h: u32, b: [u32; 4]) -> Gray {
        let mut v = vec![0u8; (w * h) as usize];
        for y in b[1]..b[3] {
            for x in b[0]..b[2] {
                v[(y * w + x) as usize] = 255;
            }
        }
        Gray::new(w, h, v).unwrap()
    }

    fn params(mode: Mode, text: &str) -> Params<'_> {
        Params { mode, text, references: &[], tier: None }
    }

    #[test]
    fn patch_is_opaque_inside_selection_transparent_far_away() {
        let img = photo(900, 700);
        let s = sel(900, 700, [300, 250, 500, 400]);
        let fake = Fake::new([250, 10, 10]);
        let out = generate(&img, &s, &params(Mode::Fill, "um sol"), &Config::default(), &fake, &|_| {}).unwrap();
        let (ox, oy) = out.origin;
        let at = |x: u32, y: u32| {
            let i = (((y - oy) * out.patch.w + (x - ox)) * 4) as usize;
            &out.patch.px[i..i + 4]
        };
        assert_eq!(at(400, 320), [250, 10, 10, 255]);
        assert_eq!(at(ox, oy)[3], 0);
        assert!(out.patch.w >= 200 && out.patch.h >= 150);
        // feathered edge: some pixel between fully transparent and opaque
        let mid = (ox..ox + out.patch.w).map(|x| at(x, 325)[3]).filter(|&a| a > 0 && a < 255).count();
        assert!(mid > 0);
    }

    #[test]
    fn request_carries_prompt_mask_size_and_references() {
        let img = photo(900, 700);
        let s = sel(900, 700, [300, 250, 500, 400]);
        let fake = Fake::new([0, 0, 0]);
        let refs = [photo(40, 30)];
        let p = Params { mode: Mode::Fill, text: "gato", references: &refs, tier: Some("media") };
        generate(&img, &s, &p, &Config::default(), &fake, &|_| {}).unwrap();
        let seen = fake.seen.lock().unwrap();
        let r = &seen[0];
        assert!(r.prompt.contains("create: gato"));
        assert!(r.prompt.contains("references"));
        assert_eq!(r.images_png.len(), 2);
        assert_eq!(r.model, "gpt-image-2.5-sunburst");
        assert_eq!(r.quality, "medium");
        assert_eq!((r.size.0 % 16, r.size.1 % 16), (0, 0));
        // mask: same size as the sent image, transparent where selected
        let mask = crate::imgops::decode_png(&r.mask_png).unwrap();
        assert_eq!((mask.w, mask.h), r.size);
        assert!(mask.px.chunks(4).any(|p| p[3] == 0));
        assert!(mask.px.chunks(4).any(|p| p[3] == 255));
    }

    #[test]
    fn remove_mode_ignores_text() {
        let img = photo(600, 600);
        let s = sel(600, 600, [100, 100, 200, 200]);
        let fake = Fake::new([1, 2, 3]);
        generate(&img, &s, &params(Mode::Remove, "ignorado"), &Config::default(), &fake, &|_| {}).unwrap();
        assert!(fake.seen.lock().unwrap()[0].prompt.starts_with("Remove whatever"));
    }

    #[test]
    fn at_most_max_references_are_sent() {
        let img = photo(600, 600);
        let s = sel(600, 600, [100, 100, 200, 200]);
        let fake = Fake::new([1, 2, 3]);
        let refs: Vec<Rgba8> = (0..9).map(|_| photo(10, 10)).collect();
        let p = Params { mode: Mode::Fill, text: "", references: &refs, tier: None };
        generate(&img, &s, &p, &Config::default(), &fake, &|_| {}).unwrap();
        assert_eq!(fake.seen.lock().unwrap()[0].images_png.len(), 1 + 4);
    }

    #[test]
    fn untouched_result_gets_a_warning() {
        struct Echo;
        impl Backend for Echo {
            fn edit(&self, req: &EditRequest) -> std::result::Result<Vec<u8>, BackendError> {
                Ok(req.images_png[0].clone())
            }
        }
        let img = photo(600, 600);
        let s = sel(600, 600, [100, 100, 300, 300]);
        let out = generate(&img, &s, &params(Mode::Fill, ""), &Config::default(), &Echo, &|_| {}).unwrap();
        assert!(out.warning.contains("quase não alterou"));
    }

    #[test]
    fn backend_errors_become_friendly_messages() {
        let img = photo(600, 600);
        let s = sel(600, 600, [100, 100, 300, 300]);
        let mut fake = Fake::new([0; 3]);
        fake.fail = Some(BackendError::Http { status: 401, detail: "nope".into(), code: String::new() });
        let e = generate(&img, &s, &params(Mode::Fill, ""), &Config::default(), &fake, &|_| {}).unwrap_err();
        assert!(e.0.contains("Chave recusada"));
        fake.fail = Some(BackendError::Other("sem internet".into()));
        assert_eq!(generate(&img, &s, &params(Mode::Fill, ""), &Config::default(), &fake, &|_| {}).unwrap_err().0, "sem internet");
    }

    #[test]
    fn mismatched_or_empty_selection_is_rejected_without_calling_the_api() {
        let fake = Fake::new([0; 3]);
        let img = photo(300, 300);
        assert!(generate(&img, &sel(200, 200, [0, 0, 10, 10]), &params(Mode::Fill, ""), &Config::default(), &fake, &|_| {}).is_err());
        assert!(generate(&img, &sel(300, 300, [0, 0, 0, 0]), &params(Mode::Fill, ""), &Config::default(), &fake, &|_| {}).is_err());
        assert!(fake.seen.lock().unwrap().is_empty());
    }

    #[test]
    fn works_on_a_tiny_image() {
        let fake = Fake::new([9, 9, 9]);
        let img = photo(40, 30);
        let out = generate(&img, &sel(40, 30, [10, 10, 20, 20]), &params(Mode::Fill, ""), &Config::default(), &fake, &|_| {}).unwrap();
        assert_eq!((out.patch.w, out.patch.h), (out.plan.ctx_size().0, out.plan.ctx_size().1));
    }
}
