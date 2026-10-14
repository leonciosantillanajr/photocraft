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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_follow_the_mode() {
        assert!(build_prompt(Mode::Remove, "ignored", false).starts_with("Remove whatever"));
        assert!(build_prompt(Mode::Fill, "  a red door ", false).starts_with("In the masked area, create: a red door. "));
        assert!(build_prompt(Mode::Fill, "", false).starts_with("Fill the masked area"));
        assert!(build_prompt(Mode::Fill, "", true).contains("references"));
        assert!(!build_prompt(Mode::Fill, "", false).contains("references"));
    }
}
