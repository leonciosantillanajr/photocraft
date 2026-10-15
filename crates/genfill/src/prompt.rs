//! The text sent with the image (OpenAI: the mask says where, the text says what).

use crate::run::Mode;

pub fn build_prompt(mode: Mode, user_text: &str, has_refs: bool) -> String {
    let user_text = user_text.trim();
    let task = match (mode, user_text.is_empty()) {
        (Mode::Remove, _) => {
            "Remove whatever is inside the masked area and reconstruct what should be behind it, continuing the surrounding background, textures and lines naturally. ".to_string()
        }
        (_, false) => format!("In the masked area, create: {user_text}. "),
        (_, true) => "Fill the masked area by seamlessly continuing the surrounding scene, as if nothing had been there. ".to_string(),
    };
    let rules = "Match the lighting, perspective, color grading, grain and sharpness of the rest of the image. Change only the masked area. Do not add text, borders, watermarks or frames.";
    let refs = if has_refs { " The additional images are references for the look of what to insert; do not copy their layout." } else { "" };
    format!("{task}{rules}{refs}")
}

/// Marker colours for services without a mask.
const MARKERS: [(&str, [u8; 3]); 4] = [("magenta", [255, 0, 255]), ("lime green", [0, 255, 0]), ("cyan", [0, 255, 255]), ("yellow", [255, 255, 0])];

/// The marker colour that appears least in `rgba` (straight RGBA8 bytes), sampled sparsely.
pub fn pick_marker(rgba: &[u8]) -> (&'static str, [u8; 3]) {
    let step = (rgba.len() / 4 / 9216).max(1);
    let mut best = (MARKERS[0], usize::MAX);
    for m in MARKERS {
        let n = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .step_by(step)
            .filter(|p| (i32::from(p[0]) - i32::from(m.1[0])).abs() + (i32::from(p[1]) - i32::from(m.1[1])).abs() + (i32::from(p[2]) - i32::from(m.1[2])).abs() < 90)
            .count();
        if n < best.1 {
            best = (m, n);
        }
    }
    best.0
}

/// The text for services without a mask: the first image has the area painted solid `marker`.
pub fn build_prompt_marked(mode: Mode, user_text: &str, marker: &str, has_refs: bool) -> String {
    let user_text = user_text.trim();
    let base = format!("The first image has a region painted solid {marker}. ");
    let task = match (mode, user_text.is_empty()) {
        (Mode::Remove, _) => "Remove whatever is in that painted region and reconstruct what should be behind it, continuing the surrounding background, textures and lines naturally. ".to_string(),
        (_, false) => format!("Replace ONLY that painted region with: {user_text}. "),
        (_, true) => "Fill that painted region by seamlessly continuing the surrounding scene, as if it had never been painted over. ".to_string(),
    };
    let rules = format!(
        "Match the lighting, perspective, color grading, grain and sharpness of the rest of the image. Do not change anything outside the {marker} region. Never leave any {marker} color in the result. Do not add text, borders, watermarks or frames. Return the whole image with the same framing and composition."
    );
    let refs = if has_refs { " The other attached images are references for the appearance of what to insert; use them for look and style, not for layout." } else { "" };
    format!("{base}{task}{rules}{refs}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_avoids_colours_in_the_picture() {
        let magenta_heavy: Vec<u8> = [255u8, 0, 255, 255].iter().copied().cycle().take(400).collect();
        assert_ne!(pick_marker(&magenta_heavy).0, "magenta");
        let grey: Vec<u8> = [90u8, 90, 90, 255].iter().copied().cycle().take(400).collect();
        assert_eq!(pick_marker(&grey).0, "magenta");
        assert_eq!(pick_marker(&[]).0, "magenta");
    }

    #[test]
    fn marked_prompts_name_the_colour() {
        let p = build_prompt_marked(Mode::Fill, " a door ", "cyan", true);
        assert!(p.starts_with("The first image has a region painted solid cyan. Replace ONLY that painted region with: a door. "));
        assert!(p.contains("Never leave any cyan color") && p.contains("references"));
        assert!(build_prompt_marked(Mode::Remove, "x", "cyan", false).contains("Remove whatever"));
        assert!(build_prompt_marked(Mode::Fill, "", "cyan", false).contains("Fill that painted region"));
    }

    #[test]
    fn prompts_follow_the_mode() {
        assert!(build_prompt(Mode::Remove, "ignored", false).starts_with("Remove whatever"));
        assert!(build_prompt(Mode::Fill, "  a red door ", false).starts_with("In the masked area, create: a red door. "));
        assert!(build_prompt(Mode::Fill, "", false).starts_with("Fill the masked area"));
        assert!(build_prompt(Mode::Fill, "", true).contains("references"));
        assert!(!build_prompt(Mode::Fill, "", false).contains("references"));
    }
}
