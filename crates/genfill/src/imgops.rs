//! Small pixel helpers on plain buffers: crop, resize, grow, blur and PNG in/out.

use crate::{GenFillError, Result};

/// Largest side accepted anywhere in this crate (guards allocations).
pub const MAX_SIDE: u32 = 32_768;

fn check_dims(w: u32, h: u32, ch: usize) -> Result<usize> {
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
        return Err(GenFillError::new(format!("Tamanho de imagem inválido: {w}×{h}.")));
    }
    (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(ch)).ok_or_else(|| GenFillError::new("Imagem grande demais."))
}

/// Straight (non-premultiplied) 8-bit RGBA, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8 {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

/// One 8-bit channel (a mask), row-major. 255 = selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gray {
    pub w: u32,
    pub h: u32,
    pub v: Vec<u8>,
}

impl Rgba8 {
    pub fn new(w: u32, h: u32, px: Vec<u8>) -> Result<Self> {
        if check_dims(w, h, 4)? != px.len() {
            return Err(GenFillError::new("Os pixels não batem com o tamanho da imagem."));
        }
        Ok(Self { w, h, px })
    }

    pub fn blank(w: u32, h: u32) -> Result<Self> {
        let n = check_dims(w, h, 4)?;
        Ok(Self { w, h, px: vec![0; n] })
    }

    /// The `[x0,x1)×[y0,y1)` window; the box must lie inside the image.
    pub fn crop(&self, b: [u32; 4]) -> Result<Self> {
        let [x0, y0, x1, y1] = b;
        if x1 <= x0 || y1 <= y0 || x1 > self.w || y1 > self.h {
            return Err(GenFillError::new("Recorte fora da imagem."));
        }
        let (cw, ch) = (x1 - x0, y1 - y0);
        let mut px = Vec::with_capacity(check_dims(cw, ch, 4)?);
        for y in y0..y1 {
            let o = (y as usize * self.w as usize + x0 as usize) * 4;
            px.extend_from_slice(self.px.get(o..o + cw as usize * 4).ok_or_else(|| GenFillError::new("Recorte fora da imagem."))?);
        }
        Ok(Self { w: cw, h: ch, px })
    }

    /// Drops alpha (the API takes opaque RGB), like Pillow's `convert("RGB")`.
    pub fn rgb_bytes(&self) -> Vec<u8> {
        self.px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
    }

    pub fn resize(&self, nw: u32, nh: u32) -> Result<Self> {
        let px = resample(&self.px, self.w, self.h, 4, nw, nh)?;
        Ok(Self { w: nw, h: nh, px })
    }

    /// Alpha-composites `top` (straight RGBA) onto `self` with its top-left corner at `(x, y)`.
    /// Parts of `top` outside the image are ignored.
    pub fn composite_over(&mut self, top: &Rgba8, x: u32, y: u32) {
        for ty in 0..top.h {
            let dy = y.saturating_add(ty);
            if dy >= self.h {
                break;
            }
            for tx in 0..top.w {
                let dx = x.saturating_add(tx);
                if dx >= self.w {
                    break;
                }
                let si = (ty as usize * top.w as usize + tx as usize) * 4;
                let di = (dy as usize * self.w as usize + dx as usize) * 4;
                let (Some(s), Some(d)) = (top.px.get(si..si + 4), self.px.get_mut(di..di + 4)) else { continue };
                let sa = s[3] as f32 / 255.0;
                if sa <= 0.0 {
                    continue;
                }
                let da = d[3] as f32 / 255.0;
                let oa = sa + da * (1.0 - sa);
                for c in 0..3 {
                    let v = (s[c] as f32 * sa + d[c] as f32 * da * (1.0 - sa)) / oa;
                    d[c] = (v + 0.5).clamp(0.0, 255.0) as u8;
                }
                d[3] = (oa * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    /// PNG bytes of the RGB channels (opaque).
    pub fn to_png_rgb(&self) -> Result<Vec<u8>> {
        encode_png(self.w, self.h, png::ColorType::Rgb, &self.rgb_bytes())
    }
}

impl Gray {
    pub fn new(w: u32, h: u32, v: Vec<u8>) -> Result<Self> {
        if check_dims(w, h, 1)? != v.len() {
            return Err(GenFillError::new("A máscara não bate com o tamanho da imagem."));
        }
        Ok(Self { w, h, v })
    }

    pub fn get(&self, x: u32, y: u32) -> u8 {
        if x >= self.w || y >= self.h {
            return 0;
        }
        self.v.get(y as usize * self.w as usize + x as usize).copied().unwrap_or(0)
    }

    /// 255 where `v > 127`, else 0.
    pub fn binarize(&self) -> Self {
        Self { w: self.w, h: self.h, v: self.v.iter().map(|&p| if p > 127 { 255 } else { 0 }).collect() }
    }

    /// Bounding box `[x0,y0,x1,y1)` of pixels `> 127`, if any.
    pub fn bbox(&self) -> Option<[u32; 4]> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        let mut any = false;
        for (y, row) in self.v.chunks_exact(self.w as usize).enumerate() {
            for (x, &p) in row.iter().enumerate() {
                if p > 127 {
                    any = true;
                    x0 = x0.min(x as u32);
                    x1 = x1.max(x as u32 + 1);
                    y0 = y0.min(y as u32);
                    y1 = y1.max(y as u32 + 1);
                }
            }
        }
        any.then_some([x0, y0, x1, y1])
    }

    pub fn count_set(&self) -> usize {
        self.v.iter().filter(|&&p| p > 127).count()
    }

    pub fn crop(&self, b: [u32; 4]) -> Result<Self> {
        let [x0, y0, x1, y1] = b;
        if x1 <= x0 || y1 <= y0 || x1 > self.w || y1 > self.h {
            return Err(GenFillError::new("Recorte fora da imagem."));
        }
        let mut v = Vec::with_capacity(check_dims(x1 - x0, y1 - y0, 1)?);
        for y in y0..y1 {
            let o = y as usize * self.w as usize + x0 as usize;
            v.extend_from_slice(self.v.get(o..o + (x1 - x0) as usize).ok_or_else(|| GenFillError::new("Recorte fora da imagem."))?);
        }
        Ok(Self { w: x1 - x0, h: y1 - y0, v })
    }

    /// Nearest-neighbour resize (keeps a binary mask binary).
    pub fn resize_nearest(&self, nw: u32, nh: u32) -> Result<Self> {
        let n = check_dims(nw, nh, 1)?;
        let mut v = Vec::with_capacity(n);
        for y in 0..nh {
            let sy = ((y as u64 * self.h as u64) / nh as u64) as u32;
            for x in 0..nw {
                let sx = ((x as u64 * self.w as u64) / nw as u64) as u32;
                v.push(self.get(sx, sy));
            }
        }
        Ok(Self { w: nw, h: nh, v })
    }

    /// Dilates by `px` (square window of side `2px+1`, Pillow's `MaxFilter`).
    pub fn grow(&self, px: u32) -> Self {
        if px == 0 {
            return self.clone();
        }
        let r = px as usize;
        let tmp = sweep(&self.v, self.w as usize, self.h as usize, r, true, |a, b| a.max(b));
        Self { w: self.w, h: self.h, v: sweep(&tmp, self.w as usize, self.h as usize, r, false, |a, b| a.max(b)) }
    }

    /// Gaussian-like blur (three box passes) with the given sigma.
    pub fn blur(&self, sigma: f32) -> Self {
        if !(sigma.is_finite() && sigma > 0.0) {
            return self.clone();
        }
        let (w, h) = (self.w as usize, self.h as usize);
        let mut cur: Vec<u8> = self.v.clone();
        for r in box_radii(sigma) {
            cur = box_blur(&cur, w, h, r, true);
            cur = box_blur(&cur, w, h, r, false);
        }
        Self { w: self.w, h: self.h, v: cur }
    }

    pub fn to_png_gray(&self) -> Result<Vec<u8>> {
        encode_png(self.w, self.h, png::ColorType::Grayscale, &self.v)
    }
}

/// Runs a 1-D window reduction of half-width `r` along rows (`horizontal`) or columns.
fn sweep(src: &[u8], w: usize, h: usize, r: usize, horizontal: bool, f: impl Fn(u8, u8) -> u8) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let (outer, inner) = if horizontal { (h, w) } else { (w, h) };
    let at = |o: usize, i: usize| if horizontal { o * w + i } else { i * w + o };
    for o in 0..outer {
        for i in 0..inner {
            let lo = i.saturating_sub(r);
            let hi = (i + r).min(inner.saturating_sub(1));
            let mut m = 0u8;
            for k in lo..=hi {
                m = f(m, src.get(at(o, k)).copied().unwrap_or(0));
            }
            if let Some(d) = out.get_mut(at(o, i)) {
                *d = m;
            }
        }
    }
    out
}

/// Box radii whose three passes approximate a Gaussian of `sigma`.
fn box_radii(sigma: f32) -> [usize; 3] {
    let ideal = ((12.0 * sigma * sigma / 3.0) + 1.0).sqrt();
    let wl = (ideal.floor() as usize) | 1;
    let wl = if wl.is_multiple_of(2) { wl.saturating_sub(1).max(1) } else { wl };
    let wu = wl + 2;
    let m = ((12.0 * sigma * sigma - 3.0 * (wl * wl) as f32 - 12.0 * wl as f32 - 9.0) / (-4.0 * wl as f32 - 4.0)).round().max(0.0) as usize;
    let rl = (wl - 1) / 2;
    let ru = (wu - 1) / 2;
    [if 0 < m { rl } else { ru }, if 1 < m { rl } else { ru }, if 2 < m { rl } else { ru }]
}

fn box_blur(src: &[u8], w: usize, h: usize, r: usize, horizontal: bool) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let (outer, inner) = if horizontal { (h, w) } else { (w, h) };
    let at = |o: usize, i: usize| if horizontal { o * w + i } else { i * w + o };
    let win = (2 * r + 1) as u32;
    for o in 0..outer {
        // Running sum with edge clamping.
        let get = |i: isize| -> u32 {
            let c = i.clamp(0, inner as isize - 1) as usize;
            src.get(at(o, c)).copied().unwrap_or(0) as u32
        };
        let mut sum: u32 = (-(r as isize)..=(r as isize)).map(get).sum();
        for i in 0..inner {
            if let Some(d) = out.get_mut(at(o, i)) {
                *d = ((sum + win / 2) / win) as u8;
            }
            sum = sum + get(i as isize + r as isize + 1) - get(i as isize - r as isize);
        }
    }
    out
}

/// Separable triangle-filter resample (antialiased when shrinking) of `ch`-channel u8 data.
pub fn resample(src: &[u8], w: u32, h: u32, ch: usize, nw: u32, nh: u32) -> Result<Vec<u8>> {
    if check_dims(w, h, ch)? != src.len() {
        return Err(GenFillError::new("Buffer de imagem inconsistente."));
    }
    let total = check_dims(nw, nh, ch)?;
    if (w, h) == (nw, nh) {
        return Ok(src.to_vec());
    }
    // Horizontal pass to f32, then vertical.
    let tmp = pass(src.iter().map(|&b| b as f32), w as usize, h as usize, ch, nw as usize, true);
    let out = pass(tmp.into_iter(), nw as usize, h as usize, ch, nh as usize, false);
    let mut bytes: Vec<u8> = out.into_iter().map(|v| (v + 0.5).clamp(0.0, 255.0) as u8).collect();
    bytes.truncate(total);
    Ok(bytes)
}

/// One resampling pass along x (`horizontal`) or y, producing `new_len` samples per line.
fn pass(src: impl Iterator<Item = f32>, w: usize, h: usize, ch: usize, new_len: usize, horizontal: bool) -> Vec<f32> {
    let data: Vec<f32> = src.collect();
    let (old_len, lines) = if horizontal { (w, h) } else { (h, w) };
    let scale = old_len as f32 / new_len as f32;
    let support = scale.max(1.0);
    let (ow, oh) = if horizontal { (new_len, h) } else { (w, new_len) };
    let mut out = vec![0f32; ow * oh * ch];
    for n in 0..new_len {
        let center = (n as f32 + 0.5) * scale;
        let lo = ((center - support).floor().max(0.0)) as usize;
        let hi = (((center + support).ceil()) as usize).min(old_len);
        let weights: Vec<(usize, f32)> = (lo..hi)
            .map(|i| (i, (1.0 - ((i as f32 + 0.5 - center) / support).abs()).max(0.0)))
            .filter(|(_, wt)| *wt > 0.0)
            .collect();
        let wsum: f32 = weights.iter().map(|(_, wt)| wt).sum();
        for line in 0..lines {
            for c in 0..ch {
                let mut acc = 0.0;
                for &(i, wt) in &weights {
                    let idx = if horizontal { (line * w + i) * ch + c } else { (i * w + line) * ch + c };
                    acc += data.get(idx).copied().unwrap_or(0.0) * wt;
                }
                let v = if wsum > 0.0 { acc / wsum } else { 0.0 };
                let oidx = if horizontal { (line * ow + n) * ch + c } else { (n * ow + line) * ch + c };
                if let Some(d) = out.get_mut(oidx) {
                    *d = v;
                }
            }
        }
    }
    out
}

pub fn encode_png(w: u32, h: u32, color: png::ColorType, data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, w, h);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut wr = enc.write_header().map_err(|e| GenFillError::new(format!("Não consegui montar o PNG: {e}")))?;
    wr.write_image_data(data).map_err(|e| GenFillError::new(format!("Não consegui montar o PNG: {e}")))?;
    wr.finish().map_err(|e| GenFillError::new(format!("Não consegui montar o PNG: {e}")))?;
    Ok(out)
}

/// Decodes a PNG into RGBA8 (gray, palette, 16-bit and RGB inputs are all converted).
pub fn decode_png(bytes: &[u8]) -> Result<Rgba8> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| GenFillError::new(format!("A imagem recebida não é um PNG válido: {e}")))?;
    let (w, h) = {
        let i = reader.info();
        (i.width, i.height)
    };
    check_dims(w, h, 4)?;
    let mut buf = vec![0; reader.output_buffer_size().ok_or_else(|| GenFillError::new("PNG grande demais."))?];
    let info = reader.next_frame(&mut buf).map_err(|e| GenFillError::new(format!("A imagem recebida não é um PNG válido: {e}")))?;
    let data = buf.get(..info.buffer_size()).ok_or_else(|| GenFillError::new("PNG inconsistente."))?;
    let px: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => data.to_vec(),
        png::ColorType::Rgb => data.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::Grayscale => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => data.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Indexed => return Err(GenFillError::new("PNG indexado inesperado.")),
    };
    Rgba8::new(w, h, px)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, p: [u8; 4]) -> Rgba8 {
        Rgba8 { w, h, px: p.iter().copied().cycle().take((w * h * 4) as usize).collect() }
    }

    #[test]
    fn crop_and_bbox() {
        let mut v = vec![0u8; 10 * 8];
        v[3 * 10 + 4] = 255;
        v[5 * 10 + 7] = 200;
        let g = Gray::new(10, 8, v).unwrap();
        assert_eq!(g.bbox(), Some([4, 3, 8, 6]));
        assert_eq!(g.crop([4, 3, 8, 6]).unwrap().count_set(), 2);
        assert!(g.crop([0, 0, 11, 2]).is_err());
        assert_eq!(Gray::new(2, 2, vec![0; 4]).unwrap().bbox(), None);
    }

    #[test]
    fn grow_is_square_window() {
        let mut v = vec![0u8; 9 * 9];
        v[4 * 9 + 4] = 255;
        let g = Gray::new(9, 9, v).unwrap().grow(2);
        assert_eq!(g.bbox(), Some([2, 2, 7, 7]));
        assert_eq!(g.count_set(), 25);
    }

    #[test]
    fn blur_keeps_flat_areas_and_softens_edges() {
        let mut v = vec![0u8; 40 * 40];
        for y in 0..40 {
            for x in 20..40 {
                v[y * 40 + x] = 255;
            }
        }
        let b = Gray::new(40, 40, v).unwrap().blur(3.0);
        assert_eq!(b.get(0, 20), 0);
        assert_eq!(b.get(39, 20), 255);
        let mid = b.get(20, 20);
        assert!(mid > 60 && mid < 200, "{mid}");
    }

    #[test]
    fn resize_preserves_flat_colour_and_size() {
        let s = solid(30, 20, [10, 200, 90, 255]);
        let up = s.resize(64, 48).unwrap();
        assert_eq!((up.w, up.h), (64, 48));
        assert!(up.px.chunks(4).all(|p| p == [10, 200, 90, 255]));
        let down = s.resize(7, 5).unwrap();
        assert!(down.px.chunks(4).all(|p| p == [10, 200, 90, 255]));
        assert!(s.resize(0, 5).is_err());
    }

    #[test]
    fn nearest_resize_stays_binary() {
        let g = Gray::new(4, 4, (0..16).map(|i| if i % 2 == 0 { 255 } else { 0 }).collect()).unwrap();
        let r = g.resize_nearest(10, 10).unwrap();
        assert!(r.v.iter().all(|&p| p == 0 || p == 255));
    }

    #[test]
    fn composite_over_blends_and_clips() {
        let mut base = solid(4, 4, [0, 0, 0, 255]);
        let top = solid(3, 3, [200, 100, 50, 255]);
        base.composite_over(&top, 2, 2); // only the 2x2 corner fits
        assert_eq!(&base.px[(2 * 4 + 2) * 4..(2 * 4 + 2) * 4 + 4], [200, 100, 50, 255]);
        assert_eq!(&base.px[0..4], [0, 0, 0, 255]);
        let half = solid(1, 1, [255, 255, 255, 128]);
        base.composite_over(&half, 0, 0);
        assert!(base.px[0] > 120 && base.px[0] < 135, "{}", base.px[0]);
        // transparent base takes the top's colour and alpha
        let mut clear = solid(1, 1, [0, 0, 0, 0]);
        clear.composite_over(&half, 0, 0);
        assert_eq!(clear.px, [255, 255, 255, 128]);
        base.composite_over(&top, 100, 100); // fully outside: no-op, no panic
    }

    #[test]
    fn png_round_trip() {
        let img = solid(5, 3, [1, 2, 3, 255]);
        let bytes = img.to_png_rgb().unwrap();
        assert_eq!(decode_png(&bytes).unwrap(), img);
        let m = Gray::new(2, 2, vec![0, 255, 255, 0]).unwrap();
        let back = decode_png(&m.to_png_gray().unwrap()).unwrap();
        assert_eq!(back.px[4], 255);
        assert!(decode_png(b"not a png").is_err());
    }
}
