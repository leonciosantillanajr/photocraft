//! Finds the empty (transparent) areas of an image, for "fill empty areas".

use crate::{Gray, Result, Rgba8};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmptyRegion {
    /// `[x0,y0,x1,y1)`
    pub bbox: [u32; 4],
    /// Number of empty pixels in the region.
    pub count: usize,
    /// Image-sized mask: 255 on this region's empty pixels.
    pub mask: Gray,
}

/// Separate empty areas become separate regions, largest first (at most `max_regions`).
/// A pixel is empty when its alpha is below `thr`. Regions are connected on a coarse grid
/// (cells of `ceil(max(w,h)/max_side)` pixels) so near-touching specks merge into one area.
pub fn find_empty_regions(image: &Rgba8, thr: u8, min_area: usize, max_side: u32, max_regions: usize) -> Result<Vec<EmptyRegion>> {
    let (w, h) = (image.w as usize, image.h as usize);
    let empty: Vec<bool> = image.px.as_chunks::<4>().0.iter().map(|p| p[3] < thr).collect();
    if !empty.iter().any(|&e| e) {
        return Ok(Vec::new());
    }
    let f = (w.max(h)).div_ceil(max_side.max(1) as usize).max(1);
    let (gw, gh) = (w.div_ceil(f), h.div_ceil(f));
    // a cell is empty if ANY of its pixels is
    let mut cell = vec![false; gw * gh];
    for y in 0..h {
        for x in 0..w {
            if empty.get(y * w + x).copied().unwrap_or(false)
                && let Some(c) = cell.get_mut((y / f) * gw + x / f)
            {
                *c = true;
            }
        }
    }
    let mut lab = vec![0u32; gw * gh];
    let mut n = 0u32;
    for start in 0..gw * gh {
        if !cell[start] || lab[start] != 0 {
            continue;
        }
        n += 1;
        lab[start] = n;
        let mut stack = vec![start];
        while let Some(c) = stack.pop() {
            let (cx, cy) = (c % gw, c / gw);
            let mut nb = [None; 4];
            if cy > 0 {
                nb[0] = Some(c - gw);
            }
            if cy + 1 < gh {
                nb[1] = Some(c + gw);
            }
            if cx > 0 {
                nb[2] = Some(c - 1);
            }
            if cx + 1 < gw {
                nb[3] = Some(c + 1);
            }
            for k in nb.into_iter().flatten() {
                if cell[k] && lab[k] == 0 {
                    lab[k] = n;
                    stack.push(k);
                }
            }
        }
    }
    // one pass: per label, bbox + count + pixels
    let mut stats: Vec<([u32; 4], usize)> = vec![([u32::MAX, u32::MAX, 0, 0], 0); n as usize];
    for y in 0..h {
        for x in 0..w {
            if !empty.get(y * w + x).copied().unwrap_or(false) {
                continue;
            }
            let l = lab.get((y / f) * gw + x / f).copied().unwrap_or(0);
            if let Some((b, c)) = stats.get_mut((l as usize).wrapping_sub(1)) {
                b[0] = b[0].min(x as u32);
                b[1] = b[1].min(y as u32);
                b[2] = b[2].max(x as u32 + 1);
                b[3] = b[3].max(y as u32 + 1);
                *c += 1;
            }
        }
    }
    let min_area = min_area.max(1);
    let mut order: Vec<usize> = (0..n as usize).filter(|&i| stats[i].1 >= min_area).collect();
    order.sort_by(|&a, &b| stats[b].1.cmp(&stats[a].1));
    order.truncate(max_regions);
    let mut out = Vec::with_capacity(order.len());
    for i in order {
        let label = i as u32 + 1;
        let mut v = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                if empty.get(y * w + x).copied().unwrap_or(false) && lab.get((y / f) * gw + x / f) == Some(&label) {
                    v[y * w + x] = 255;
                }
            }
        }
        out.push(EmptyRegion { bbox: stats[i].0, count: stats[i].1, mask: Gray::new(image.w, image.h, v)? });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, holes: &[[u32; 4]]) -> Rgba8 {
        let mut px = vec![255u8; (w * h * 4) as usize];
        for b in holes {
            for y in b[1]..b[3] {
                for x in b[0]..b[2] {
                    px[((y * w + x) * 4 + 3) as usize] = 0;
                }
            }
        }
        Rgba8::new(w, h, px).unwrap()
    }

    #[test]
    fn opaque_image_has_no_regions() {
        assert!(find_empty_regions(&img(50, 50, &[]), 16, 0, 800, 8).unwrap().is_empty());
    }

    #[test]
    fn separate_holes_become_regions_largest_first() {
        let r = find_empty_regions(&img(100, 80, &[[0, 0, 10, 10], [50, 40, 90, 70]]), 16, 0, 800, 8).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].bbox, [50, 40, 90, 70]);
        assert_eq!(r[0].count, 40 * 30);
        assert_eq!(r[1].bbox, [0, 0, 10, 10]);
        assert_eq!(r[0].mask.count_set(), 1200);
    }

    #[test]
    fn limits_and_min_area_apply() {
        let holes: Vec<[u32; 4]> = (0..10).map(|i| [i * 10, 0, i * 10 + 4 + (i % 3), 4]).collect();
        let r = find_empty_regions(&img(100, 10, &holes), 16, 0, 800, 3).unwrap();
        assert_eq!(r.len(), 3);
        let big = find_empty_regions(&img(100, 10, &holes), 16, 20, 800, 8).unwrap();
        assert!(big.iter().all(|e| e.count >= 20));
    }

    #[test]
    fn coarse_grid_merges_nearby_specks() {
        // 2 holes 1px apart merge when the grid cell is 2px (max_side = half the size)
        let r = find_empty_regions(&img(100, 10, &[[10, 0, 14, 4], [15, 0, 19, 4]]), 16, 0, 50, 8).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].count, 32);
    }
}
