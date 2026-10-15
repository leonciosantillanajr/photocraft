//! The floating Generative Fill bar over the canvas (Adobe's contextual task bar).
//!
//! With a selection it offers **Generative Fill** (opens a prompt), **Remove** and **Fill Empty
//! Areas**, plus **Deselect**. The prompt state has a text field, a reference-image button, Cancel
//! and Generate. While the engine works the bar shows the elapsed time and Cancel. The grip on the
//! left drags the bar; where it sits is remembered (`UiState::genfill_bar_pos`).
//!
//! The bar only starts the engine's `edit.generative*` commands (through [`crate::jobs_ui::run`]):
//! the work runs as a cancellable background job and lands as new layers in one undo step.

use egui::{Align2, Color32, CursorIcon, Key, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Ui, vec2};
use photocraft_engine::jobs::JobId;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

const BAR_H: f32 = 34.0;
const DEFAULT_POS: [f32; 2] = [0.5, 56.0];
const MAX_REFS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Idle,
    Prompt,
    NeedKey,
    Busy(JobId),
}

/// The bar's own state (not saved; only the position is).
#[derive(Default)]
pub struct State {
    phase: Phase,
    prompt: String,
    refs: Vec<String>,
    key_input: String,
    doc: Option<photocraft_doc::DocId>,
    focus_field: bool,
    key_synced: bool,
}

impl State {
    fn reset(&mut self) {
        self.phase = Phase::Idle;
        self.focus_field = false;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Icon {
    Sparkle,
    Eraser,
    Empty,
    Pin,
    Image,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Look {
    Primary,
    Plain,
    Outline,
    Ghost,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Act {
    OpenPrompt,
    Generate,
    Remove,
    FillEmpty,
    Deselect,
    BackToIdle,
    Cancel(JobId),
    TogglePin,
    PickRef,
    ClearRefs,
    SaveKey,
}

fn field_id() -> egui::Id {
    egui::Id::new("genfill-prompt")
}

/// A pill button: optional icon, optional label. Returns the click response.
fn pill(ui: &mut Ui, t: &Tokens, icon: Option<Icon>, label: &str, look: Look, tip: &str, badge: usize) -> egui::Response {
    let font = crate::theme::semibold(12.5);
    let text_color = match look {
        Look::Primary => t.primary_text,
        Look::Ghost => t.text_dim,
        _ => t.text,
    };
    let galley = (!label.is_empty()).then(|| ui.painter().layout_no_wrap(label.to_string(), font, text_color));
    let icon_w = if icon.is_some() { 16.0 } else { 0.0 };
    let gap = if icon.is_some() && galley.is_some() { 6.0 } else { 0.0 };
    let text_w = galley.as_ref().map_or(0.0, |g| g.size().x);
    let pad = if galley.is_some() { 12.0 } else { 8.0 };
    let size = vec2(pad * 2.0 + icon_w + gap + text_w, BAR_H - 6.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let (fill, stroke) = match look {
        Look::Primary => (if resp.is_pointer_button_down_on() { t.primary_bg.gamma_multiply(0.8) } else if resp.hovered() { t.primary_bg.lerp_to_gamma(t.primary_text, 0.14) } else { t.primary_bg }, Stroke::NONE),
        Look::Outline => (if resp.hovered() { t.hover } else { Color32::TRANSPARENT }, Stroke::new(1.0, t.field_border)),
        _ => (if resp.is_pointer_button_down_on() { t.pressed } else if resp.hovered() { t.hover } else { Color32::TRANSPARENT }, Stroke::NONE),
    };
    let radius = rect.height() / 2.0;
    ui.painter().rect_filled(rect, radius, fill);
    if stroke != Stroke::NONE {
        ui.painter().rect_stroke(rect, radius, stroke, StrokeKind::Inside);
    }
    let mut x = rect.min.x + pad;
    if let Some(i) = icon {
        let c = Pos2::new(x + icon_w / 2.0, rect.center().y);
        let color = if look == Look::Primary { t.primary_text } else { t.icon };
        paint_icon(ui, i, c, color);
        x += icon_w + gap;
    }
    if let Some(g) = galley {
        let pos = Pos2::new(x, rect.center().y - g.size().y / 2.0);
        ui.painter().galley(pos, g, text_color);
    }
    if badge > 0 {
        let c = Pos2::new(rect.max.x - 5.0, rect.min.y + 5.0);
        ui.painter().circle_filled(c, 6.5, t.primary_bg);
        ui.painter().text(c, Align2::CENTER_CENTER, badge.to_string(), egui::FontId::proportional(9.5), t.primary_text);
    }
    if tip.is_empty() { resp } else { resp.on_hover_text(tip).on_hover_cursor(CursorIcon::PointingHand) }
}

fn rhombus(c: Pos2, rx: f32, ry: f32) -> Vec<Pos2> {
    vec![Pos2::new(c.x - rx, c.y), Pos2::new(c.x, c.y - ry), Pos2::new(c.x + rx, c.y), Pos2::new(c.x, c.y + ry)]
}

fn star(ui: &Ui, c: Pos2, r: f32, color: Color32) {
    let p = ui.painter();
    p.add(Shape::convex_polygon(rhombus(c, r, r * 0.3), color, Stroke::NONE));
    p.add(Shape::convex_polygon(rhombus(c, r * 0.3, r), color, Stroke::NONE));
}

fn paint_icon(ui: &Ui, icon: Icon, c: Pos2, color: Color32) {
    let p = ui.painter();
    let s = Stroke::new(1.5, color);
    match icon {
        Icon::Sparkle => {
            star(ui, c + vec2(-1.5, 1.5), 6.0, color);
            star(ui, c + vec2(4.5, -4.5), 3.2, color);
        }
        Icon::Eraser => {
            // a slanted block with a band, sitting on a baseline
            let q = vec![c + vec2(-2.0, -6.0), c + vec2(6.5, -2.0), c + vec2(1.5, 4.5), c + vec2(-7.0, 0.5)];
            p.add(Shape::convex_polygon(q, Color32::TRANSPARENT, s));
            p.line_segment([c + vec2(-3.0, 2.8), c + vec2(3.0, -2.6)], Stroke::new(1.2, color));
            p.line_segment([c + vec2(-3.0, 7.0), c + vec2(7.0, 7.0)], s);
        }
        Icon::Empty => {
            // dashed square with a plus
            let r = Rect::from_center_size(c, vec2(14.0, 14.0));
            for (a, b) in [
                (r.left_top(), r.left_top() + vec2(4.0, 0.0)),
                (r.left_top(), r.left_top() + vec2(0.0, 4.0)),
                (r.right_top(), r.right_top() + vec2(-4.0, 0.0)),
                (r.right_top(), r.right_top() + vec2(0.0, 4.0)),
                (r.left_bottom(), r.left_bottom() + vec2(4.0, 0.0)),
                (r.left_bottom(), r.left_bottom() + vec2(0.0, -4.0)),
                (r.right_bottom(), r.right_bottom() + vec2(-4.0, 0.0)),
                (r.right_bottom(), r.right_bottom() + vec2(0.0, -4.0)),
            ] {
                p.line_segment([a, b], s);
            }
            p.line_segment([c + vec2(-3.0, 0.0), c + vec2(3.0, 0.0)], s);
            p.line_segment([c + vec2(0.0, -3.0), c + vec2(0.0, 3.0)], s);
        }
        Icon::Pin => {
            // a push-pin: round head, short body, needle
            p.circle_stroke(c + vec2(0.0, -3.0), 3.6, s);
            p.line_segment([c + vec2(0.0, 0.6), c + vec2(0.0, 7.0)], s);
            p.line_segment([c + vec2(-3.5, 0.6), c + vec2(3.5, 0.6)], s);
        }
        Icon::Image => {
            let r = Rect::from_center_size(c, vec2(15.0, 12.0));
            p.rect_stroke(r, 2.0, s, StrokeKind::Inside);
            p.add(Shape::line(vec![r.left_bottom() + vec2(1.5, -1.5), c + vec2(-2.0, -1.0), c + vec2(1.0, 2.0), c + vec2(3.0, 0.0), r.right_bottom() + vec2(-1.0, -2.0)], Stroke::new(1.2, color)));
            p.circle_filled(r.left_top() + vec2(4.5, 3.8), 1.5, color);
        }
    }
}

fn grip(ui: &mut Ui, t: &Tokens) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(16.0, BAR_H - 6.0), Sense::drag());
    let color = if resp.hovered() || resp.dragged() { t.text_dim } else { t.text_faint };
    for col in 0..2 {
        for row in 0..3 {
            let c = rect.center() + vec2(-2.5 + col as f32 * 5.0, -5.0 + row as f32 * 5.0);
            ui.painter().circle_filled(c, 1.3, color);
        }
    }
    resp.on_hover_cursor(CursorIcon::Grab)
}

fn sep(ui: &mut Ui, t: &Tokens) {
    let (r, _) = ui.allocate_exact_size(vec2(9.0, BAR_H - 12.0), Sense::hover());
    ui.painter().line_segment([r.center_top(), r.center_bottom()], Stroke::new(1.0, t.separator));
}

fn file_names(refs: &[String]) -> String {
    refs.iter().map(|p| std::path::Path::new(p).file_name().and_then(|n| n.to_str()).unwrap_or(p.as_str()).to_string()).collect::<Vec<_>>().join("\n")
}

/// Draws the bar (when it applies) and performs what was clicked. Call once per frame, after the canvas.
pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !app.genfill.key_synced {
        photocraft_engine::genfill_cmds::set_api_key(&app.ui.genfill_api_key);
        app.genfill.key_synced = true;
    }
    let Some(d) = app.session.active() else {
        app.genfill.reset();
        return;
    };
    let (doc_id, has_selection) = (d.doc.id, d.doc.selection.is_some());
    if app.genfill.doc != Some(doc_id) {
        app.genfill.doc = Some(doc_id);
        app.genfill.reset();
    }
    if let Phase::Busy(j) = app.genfill.phase
        && app.session.job(j).is_none()
    {
        app.genfill.phase = Phase::Idle;
    }
    let busy = matches!(app.genfill.phase, Phase::Busy(_));
    let pinned = app.ui.genfill_bar_pinned;
    if !has_selection && matches!(app.genfill.phase, Phase::Prompt) {
        app.genfill.phase = Phase::Idle;
    }
    if !has_selection && !busy && !pinned {
        app.genfill.reset();
        return;
    }

    let t = Tokens::get(ctx);
    let canvas = app.last_canvas_rect;
    let [fx, off] = app.ui.genfill_bar_pos.unwrap_or(DEFAULT_POS);
    let max_off = (canvas.height() - BAR_H - 8.0).max(8.0);
    let anchor = Pos2::new(canvas.min.x + fx.clamp(0.0, 1.0) * canvas.width(), canvas.max.y - off.clamp(8.0, max_off));
    let phase = app.genfill.phase;
    let elapsed = match phase {
        Phase::Busy(j) => app.session.job(j).map_or(0.0, |i| i.elapsed_ms / 1000.0),
        _ => 0.0,
    };

    let mut act: Option<Act> = None;
    let mut drag = vec2(0.0, 0.0);
    let focus = std::mem::take(&mut app.genfill.focus_field);
    egui::Area::new(egui::Id::new("genfill-bar")).order(egui::Order::Foreground).pivot(Align2::CENTER_BOTTOM).fixed_pos(anchor).interactable(true).show(ctx, |ui| {
        egui::Frame::NONE
            .fill(t.card)
            .stroke(Stroke::new(1.0, t.card_border))
            .corner_radius(BAR_H / 2.0 + 3.0)
            .inner_margin(egui::Margin::symmetric(8, 3))
            .shadow(egui::Shadow { offset: [0, 3], blur: 14, spread: 0, color: t.shadow })
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = vec2(4.0, 0.0);
                ui.horizontal_centered(|ui| {
                    let g = grip(ui, &t);
                    if g.dragged() {
                        drag = g.drag_delta();
                    }
                    match phase {
                        Phase::Idle => {
                            if has_selection {
                                if pill(ui, &t, Some(Icon::Sparkle), tl!("Generative Fill"), Look::Primary, tl!("Describe what to create inside the selection"), 0).clicked() {
                                    act = Some(Act::OpenPrompt);
                                }
                                if pill(ui, &t, Some(Icon::Eraser), tl!("Remove"), Look::Plain, tl!("Remove what is selected and rebuild the background"), 0).clicked() {
                                    act = Some(Act::Remove);
                                }
                            }
                            if pill(ui, &t, Some(Icon::Empty), tl!("Fill empty areas"), if has_selection { Look::Plain } else { Look::Primary }, tl!("Fill the transparent areas of the image"), 0).clicked() {
                                act = Some(Act::FillEmpty);
                            }
                            sep(ui, &t);
                            if has_selection && pill(ui, &t, None, tl!("Cancel selection"), Look::Outline, "", 0).clicked() {
                                act = Some(Act::Deselect);
                            }
                            let tip = if pinned { tl!("Bar always visible (click to show only with a selection)") } else { tl!("Keep this bar always visible") };
                            if pill(ui, &t, Some(Icon::Pin), "", if pinned { Look::Outline } else { Look::Ghost }, tip, 0).clicked() {
                                act = Some(Act::TogglePin);
                            }
                        }
                        Phase::Prompt => {
                            let r = crate::widgets::text_field_ex(ui, field_id(), &mut app.genfill.prompt, tl!("Describe what to create (optional)"), 320.0, false, Some(4000));
                            if focus {
                                r.request_focus();
                            }
                            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                            let esc = r.has_focus() && ui.input(|i| i.key_pressed(Key::Escape));
                            if enter {
                                act = Some(Act::Generate);
                            } else if esc {
                                act = Some(Act::BackToIdle);
                            }
                            let n = app.genfill.refs.len();
                            let tip = if n == 0 { tl!("Add reference images").to_string() } else { file_names(&app.genfill.refs) };
                            if pill(ui, &t, Some(Icon::Image), "", Look::Plain, &tip, n).clicked() && n < MAX_REFS {
                                act = Some(Act::PickRef);
                            }
                            if n > 0 && pill(ui, &t, None, "×", Look::Ghost, tl!("Remove reference images"), 0).clicked() {
                                act = Some(Act::ClearRefs);
                            }
                            sep(ui, &t);
                            if pill(ui, &t, None, tl!("Cancel"), Look::Ghost, "", 0).clicked() {
                                act = Some(Act::BackToIdle);
                            }
                            if pill(ui, &t, Some(Icon::Sparkle), tl!("Generate"), Look::Primary, "", 0).clicked() {
                                act = Some(Act::Generate);
                            }
                        }
                        Phase::NeedKey => {
                            let r = crate::widgets::text_field_ex(ui, egui::Id::new("genfill-key"), &mut app.genfill.key_input, tl!("Paste your OpenAI API key (sk-…)"), 320.0, true, Some(300));
                            if focus {
                                r.request_focus();
                            }
                            if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                                act = Some(Act::SaveKey);
                            }
                            sep(ui, &t);
                            if pill(ui, &t, None, tl!("Cancel"), Look::Ghost, "", 0).clicked() {
                                act = Some(Act::BackToIdle);
                            }
                            if pill(ui, &t, None, tl!("Save key"), Look::Primary, "", 0).clicked() {
                                act = Some(Act::SaveKey);
                            }
                        }
                        Phase::Busy(job) => {
                            ui.add(egui::Spinner::new().size(15.0).color(t.accent));
                            ui.add_space(4.0);
                            let text = crate::i18n::fmt(tl!("Generating… {s} s"), &[("s", &format!("{elapsed:.0}"))]);
                            ui.label(egui::RichText::new(text).color(t.text));
                            ui.add_space(8.0);
                            if pill(ui, &t, None, tl!("Cancel"), Look::Outline, "", 0).clicked() {
                                act = Some(Act::Cancel(job));
                            }
                        }
                    }
                });
            });
    });
    if matches!(phase, Phase::Busy(_)) {
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
    }
    if drag != vec2(0.0, 0.0) {
        let w = canvas.width().max(1.0);
        app.ui.genfill_bar_pos = Some([(fx + drag.x / w).clamp(0.0, 1.0), (off - drag.y).clamp(8.0, max_off)]);
    }
    if let Some(a) = act {
        perform(app, a);
    }
}

fn start(app: &mut PhotocraftApp, command: &str, params: Value) {
    match crate::jobs_ui::run(app, command, params) {
        Ok(v) => {
            app.genfill.phase = match v.get("job").and_then(Value::as_u64) {
                Some(id) => Phase::Busy(JobId(id)),
                None => Phase::Idle, // it ran to completion inline
            };
        }
        Err(e) if e.contains("OPENAI_API_KEY") => {
            app.genfill.phase = Phase::NeedKey;
            app.genfill.focus_field = true;
        }
        Err(e) => {
            app.genfill.phase = Phase::Idle;
            crate::notices::error(app, e);
        }
    }
}

fn perform(app: &mut PhotocraftApp, act: Act) {
    match act {
        Act::OpenPrompt => {
            app.genfill.phase = Phase::Prompt;
            app.genfill.focus_field = true;
        }
        Act::BackToIdle => app.genfill.phase = Phase::Idle,
        Act::TogglePin => app.ui.genfill_bar_pinned = !app.ui.genfill_bar_pinned,
        Act::Generate => {
            let params = json!({"prompt": app.genfill.prompt.trim(), "references": app.genfill.refs});
            start(app, "edit.generativeFill", params);
            if matches!(app.genfill.phase, Phase::Busy(_) | Phase::Idle) {
                app.genfill.prompt.clear();
            }
        }
        Act::Remove => start(app, "edit.generativeRemove", json!({})),
        Act::FillEmpty => start(app, "edit.generativeFillEmpty", json!({})),
        Act::Deselect => {
            if let Err(e) = crate::jobs_ui::run(app, "select.deselect", json!({})) {
                crate::notices::error(app, e);
            }
        }
        Act::Cancel(job) => {
            crate::jobs_ui::cancel(app, job);
        }
        Act::ClearRefs => app.genfill.refs.clear(),
        Act::PickRef => {
            let room = MAX_REFS.saturating_sub(app.genfill.refs.len());
            let req = crate::FileDialogRequest::Open {
                multiple: true,
                initial_dir: None,
                extensions: Some(["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif"].iter().map(|s| s.to_string()).collect()),
            };
            // A cancelled dialog is not an error; a missing dialog service (tests, web) just does nothing.
            let _ = app.ask_file(req, move |app, answer| {
                if let crate::FileDialogAnswer::Paths(paths) = answer {
                    app.genfill.refs.extend(paths.into_iter().take(room));
                }
                Ok(json!({}))
            });
        }
        Act::SaveKey => {
            let key = app.genfill.key_input.trim().to_string();
            if key.starts_with("sk-") {
                photocraft_engine::genfill_cmds::set_api_key(&key);
                app.ui.genfill_api_key = key;
                app.genfill.key_input.clear();
                let sel = app.session.active().is_some_and(|d| d.doc.selection.is_some());
                app.genfill.phase = if sel { Phase::Prompt } else { Phase::Idle };
                app.genfill.focus_field = sel;
            } else {
                crate::notices::error(app, tl!("That does not look like an OpenAI API key (they start with sk-).").to_string());
            }
        }
    }
}
