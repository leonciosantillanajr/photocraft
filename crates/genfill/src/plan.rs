//! Which crop to send, at what size and in which quality tier (port of `make_plan`).

use crate::{GenFillError, Gray, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Tier {
    pub key: &'static str,
    pub label: &'static str,
    /// Largest context crop side (original pixels) this tier handles.
    pub max_context_px: u32,
    pub model: String,
    pub quality: &'static str,
    /// Longest side of the image sent to the API.
    pub send_px: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Context around the selection (0.5 = half the selection size on each side).
    pub context_margin: f64,
    pub min_context_px: u32,
    pub mask_grow_px: u32,
    pub feather_ratio: f64,
    pub max_references: usize,
    pub reference_max_side_px: u32,
    pub api_base: String,
    pub timeout_seconds: u64,
    pub moderation: Option<String>,
    pub size_multiple: u32,
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub aspect_ratios: Vec<(u32, u32)>,
    /// Cheapest first.
    pub tiers: Vec<Tier>,
}

impl Default for Config {
    fn default() -> Self {
        let t = |key, label, max_context_px, model: &str, quality, send_px| Tier { key, label, max_context_px, model: model.into(), quality, send_px };
        Self {
            context_margin: 0.5,
            min_context_px: 512,
            mask_grow_px: 3,
            feather_ratio: 0.006,
            max_references: 4,
            reference_max_side_px: 1024,
            api_base: "https://api.openai.com/v1".into(),
            timeout_seconds: 300,
            moderation: Some("auto".into()),
            size_multiple: 16,
            min_pixels: 720_000,
            max_pixels: 8_000_000,
            aspect_ratios: vec![(1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (9, 16), (16, 9)],
            tiers: vec![
                t("baixa", "Baixa", 1024, "gpt-image-2.5-flare", "low", 1024),
                t("media", "Média", 2048, "gpt-image-2.5-sunburst", "medium", 1536),
                t("alta", "Alta", 100_000, "gpt-image-2.5-sunburst", "high", 2048),
            ],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Selection bounding box `[x0,y0,x1,y1)` in image pixels.
    pub sel_box: [u32; 4],
    /// Context crop sent to the API, in image pixels.
    pub ctx_box: [u32; 4],
    pub tier: Tier,
    pub auto_tier: bool,
    pub aspect: (u32, u32),
    pub exact_ratio: bool,
    pub send_size: (u32, u32),
}

impl Plan {
    pub fn ctx_size(&self) -> (u32, u32) {
        (self.ctx_box[2] - self.ctx_box[0], self.ctx_box[3] - self.ctx_box[1])
    }

    pub fn describe(&self) -> String {
        let (sw, sh) = (self.sel_box[2] - self.sel_box[0], self.sel_box[3] - self.sel_box[1]);
        let (cw, ch) = self.ctx_size();
        format!(
            "Seleção {sw}×{sh} px · recorte {cw}×{ch} px · qualidade {} ({}) · {} · {} · envio {}×{}",
            self.tier.label,
            if self.auto_tier { "automática" } else { "manual" },
            self.tier.model,
            self.tier.quality,
            self.send_size.0,
            self.send_size.1
        )
    }
}

/// Moves the box inside the image without resizing it (shrinks only if it cannot fit).
fn clamp_box(b: [i64; 4], iw: i64, ih: i64) -> [i64; 4] {
    let w = (b[2] - b[0]).min(iw);
    let h = (b[3] - b[1]).min(ih);
    let x0 = b[0].min(iw - w).max(0);
    let y0 = b[1].min(ih - h).max(0);
    [x0, y0, x0 + w, y0 + h]
}

fn grow_box_to(b: [i64; 4], nw: i64, nh: i64, iw: i64, ih: i64) -> [i64; 4] {
    let cx = (b[0] + b[2]) as f64 / 2.0;
    let cy = (b[1] + b[3]) as f64 / 2.0;
    let x0 = (cx - nw as f64 / 2.0).round() as i64;
    let y0 = (cy - nh as f64 / 2.0).round() as i64;
    clamp_box([x0, y0, x0 + nw, y0 + nh], iw, ih)
}

fn nearest_ratio(w: i64, h: i64, ratios: &[(u32, u32)]) -> ((u32, u32), f64) {
    let target = w as f64 / h as f64;
    let mut best = (ratios.first().copied().unwrap_or((1, 1)), f64::MAX);
    for &(a, b) in ratios {
        if a == 0 || b == 0 {
            continue;
        }
        let err = (target / (a as f64 / b as f64)).ln().abs();
        if err < best.1 {
            best = ((a, b), err);
        }
    }
    (best.0, best.0 .0 as f64 / best.0 .1.max(1) as f64)
}

/// The size the crop is sent at: fits the tier's `send_px`, the API's pixel budget and a size multiple.
pub fn send_size(w: u32, h: u32, tier: &Tier, cfg: &Config) -> (u32, u32) {
    let (wf, hf) = (w as f64, h as f64);
    let mut scale = (tier.send_px as f64 / wf.max(hf)).min(1.0);
    let area = wf * hf * scale * scale;
    if cfg.min_pixels > 0 && area < cfg.min_pixels as f64 {
        scale *= (cfg.min_pixels as f64 / area).sqrt();
    }
    let area = wf * hf * scale * scale;
    if cfg.max_pixels > 0 && area > cfg.max_pixels as f64 {
        scale *= (cfg.max_pixels as f64 / area).sqrt();
    }
    let m = cfg.size_multiple.max(1) as f64;
    if cfg.size_multiple > 1 {
        let r = |v: f64| ((v * scale / m).round() * m).max(m) as u32;
        (r(wf), r(hf))
    } else {
        ((wf * scale).round().max(1.0) as u32, (hf * scale).round().max(1.0) as u32)
    }
}

/// Plans the request for the selection `mask` (same size as the image).
pub fn make_plan(mask: &Gray, cfg: &Config, forced_tier: Option<&str>) -> Result<Plan> {
    let sel = mask.bbox().ok_or_else(|| GenFillError::new("Não há nada selecionado. Selecione a área primeiro."))?;
    let (iw, ih) = (mask.w as i64, mask.h as i64);
    let (sw, sh) = ((sel[2] - sel[0]) as i64, (sel[3] - sel[1]) as i64);

    // 1) context around the selection
    let pad = ((sw.max(sh) as f64 * cfg.context_margin) as i64).max(32);
    let mut b = clamp_box([sel[0] as i64 - pad, sel[1] as i64 - pad, sel[2] as i64 + pad, sel[3] as i64 + pad], iw, ih);

    // 2) minimum size, so the model has enough context
    let min_px = (cfg.min_context_px as i64).min(iw).min(ih);
    let (w, h) = (b[2] - b[0], b[3] - b[1]);
    if w.max(h) < min_px {
        let f = min_px as f64 / w.max(h) as f64;
        b = grow_box_to(b, iw.min((w as f64 * f).round() as i64), ih.min((h as f64 * f).round() as i64), iw, ih);
    }

    // 3) snap to an accepted aspect ratio
    let (w, h) = (b[2] - b[0], b[3] - b[1]);
    let (aspect, ratio) = nearest_ratio(w, h, &cfg.aspect_ratios);
    let (nw, nh) = if (w as f64 / h as f64) < ratio { ((h as f64 * ratio).round() as i64, h) } else { (w, (w as f64 / ratio).round() as i64) };
    let nb = grow_box_to(b, nw, nh, iw, ih);
    let (gw, gh) = (nb[2] - nb[0], nb[3] - nb[1]);
    let exact = ((gw as f64 / gh as f64) / ratio - 1.0).abs() < 0.01;
    if exact || (gw >= w && gh >= h) {
        b = nb;
    }
    let (w, h) = (b[2] - b[0], b[3] - b[1]);

    // 4) quality by crop size
    let long = w.max(h) as u32;
    let by_size = || cfg.tiers.iter().find(|t| long <= t.max_context_px).or(cfg.tiers.last());
    let (tier, auto) = match forced_tier.and_then(|k| cfg.tiers.iter().find(|t| t.key == k)) {
        Some(t) => (t, false),
        None => (by_size().ok_or_else(|| GenFillError::new("Nenhuma faixa de qualidade configurada."))?, true),
    };
    let ctx_box = [b[0] as u32, b[1] as u32, b[2] as u32, b[3] as u32];
    Ok(Plan {
        sel_box: sel,
        ctx_box,
        tier: tier.clone(),
        auto_tier: auto,
        aspect,
        exact_ratio: exact,
        send_size: send_size(w as u32, h as u32, tier, cfg),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(w: u32, h: u32, b: [u32; 4]) -> Gray {
        let mut v = vec![0u8; (w * h) as usize];
        for y in b[1]..b[3] {
            for x in b[0]..b[2] {
                v[(y * w + x) as usize] = 255;
            }
        }
        Gray::new(w, h, v).unwrap()
    }

    #[test]
    fn empty_selection_is_a_friendly_error() {
        let e = make_plan(&mask(100, 100, [0, 0, 0, 0]), &Config::default(), None).unwrap_err();
        assert!(e.0.contains("nada selecionado"));
    }

    #[test]
    fn crop_has_margin_stays_inside_and_contains_selection() {
        let p = make_plan(&mask(4000, 3000, [1000, 1000, 1400, 1300]), &Config::default(), None).unwrap();
        let c = p.ctx_box;
        assert!(c[0] <= 1000 && c[1] <= 1000 && c[2] >= 1400 && c[3] >= 1300, "{c:?}");
        assert!(c[2] <= 4000 && c[3] <= 3000);
        assert!(p.ctx_size().0.max(p.ctx_size().1) >= 512);
    }

    #[test]
    fn selection_at_corner_is_clamped_not_cropped_away() {
        let p = make_plan(&mask(2000, 2000, [0, 0, 100, 100]), &Config::default(), None).unwrap();
        assert_eq!((p.ctx_box[0], p.ctx_box[1]), (0, 0));
        assert!(p.ctx_box[2] >= 100 && p.ctx_box[3] >= 100);
    }

    #[test]
    fn tier_follows_crop_size_and_can_be_forced() {
        let cfg = Config::default();
        let small = make_plan(&mask(4000, 4000, [100, 100, 200, 200]), &cfg, None).unwrap();
        assert_eq!(small.tier.key, "baixa");
        let big = make_plan(&mask(6000, 6000, [500, 500, 3500, 3500]), &cfg, None).unwrap();
        assert_eq!(big.tier.key, "alta");
        assert!(big.auto_tier);
        let forced = make_plan(&mask(4000, 4000, [100, 100, 200, 200]), &cfg, Some("alta")).unwrap();
        assert_eq!((forced.tier.key, forced.auto_tier), ("alta", false));
        let unknown = make_plan(&mask(4000, 4000, [100, 100, 200, 200]), &cfg, Some("nada")).unwrap();
        assert!(unknown.auto_tier);
    }

    #[test]
    fn send_size_respects_multiple_and_pixel_budget() {
        let cfg = Config::default();
        for (w, h) in [(512, 512), (1000, 700), (3000, 2000), (900, 1600)] {
            let t = cfg.tiers.last().unwrap();
            let (sw, sh) = send_size(w, h, t, &cfg);
            assert_eq!((sw % 16, sh % 16), (0, 0), "{w}x{h} -> {sw}x{sh}");
            assert!(u64::from(sw) * u64::from(sh) <= cfg.max_pixels + 16 * 4096);
            assert!(u64::from(sw) * u64::from(sh) >= cfg.min_pixels - 16 * 4096, "{w}x{h} -> {sw}x{sh}");
        }
    }

    #[test]
    fn tiny_image_does_not_panic() {
        let p = make_plan(&mask(20, 10, [2, 2, 5, 5]), &Config::default(), None).unwrap();
        assert!(p.ctx_box[2] <= 20 && p.ctx_box[3] <= 10);
    }
}
