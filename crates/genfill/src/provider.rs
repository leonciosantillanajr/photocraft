//! The image generators the user can pick, and the differences between them.

use crate::{GenFillError, Result};

/// Which service makes the image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Provider {
    /// OpenAI GPT Image 2 (accepts a mask).
    #[default]
    OpenAi,
    /// Google Nano Banana 2 (`gemini-3.1-flash-image`; no mask: the area is painted with a marker colour).
    Gemini,
    /// Meta Muse Image (OpenAI-style endpoint; no mask: the area is painted with a marker colour).
    Muse,
}

impl Provider {
    pub const ALL: [Provider; 3] = [Provider::OpenAi, Provider::Gemini, Provider::Muse];

    /// Stable id used in settings and command parameters.
    pub fn id(self) -> &'static str {
        match self {
            Provider::OpenAi => "openai",
            Provider::Gemini => "gemini",
            Provider::Muse => "muse",
        }
    }

    /// Reads an id (case-insensitive, a few aliases). `None` for anything else.
    pub fn parse(text: &str) -> Option<Provider> {
        match text.trim().to_lowercase().as_str() {
            "openai" | "gpt" | "gpt-image-2" | "gptimage2" => Some(Provider::OpenAi),
            "gemini" | "google" | "nano-banana" | "nanobanana" | "nano banana 2" => Some(Provider::Gemini),
            "muse" | "meta" | "muse-image" | "meta muse image" => Some(Provider::Muse),
            _ => None,
        }
    }

    /// Name shown in the interface.
    pub fn label(self) -> &'static str {
        match self {
            Provider::OpenAi => "GPT Image 2",
            Provider::Gemini => "Nano Banana 2",
            Provider::Muse => "Meta Muse Image",
        }
    }

    /// Company name, for messages.
    pub fn company(self) -> &'static str {
        match self {
            Provider::OpenAi => "OpenAI",
            Provider::Gemini => "Google",
            Provider::Muse => "Meta",
        }
    }

    /// Environment variables that can hold the key, first one wins.
    pub fn env_vars(self) -> &'static [&'static str] {
        match self {
            Provider::OpenAi => &["OPENAI_API_KEY"],
            Provider::Gemini => &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            Provider::Muse => &["MODEL_API_KEY", "META_API_KEY"],
        }
    }

    /// The variable named in "key not found" messages (the UI looks for it to ask for the key).
    pub fn env_var(self) -> &'static str {
        self.env_vars().first().copied().unwrap_or("API_KEY")
    }

    /// Does the service take a mask image? When not, the selected area is painted with a marker colour.
    pub fn uses_mask(self) -> bool {
        matches!(self, Provider::OpenAi)
    }

    /// Fixed model id (OpenAI picks the model by quality tier instead).
    pub fn model(self) -> Option<&'static str> {
        match self {
            Provider::OpenAi => None,
            Provider::Gemini => Some("gemini-3.1-flash-image"),
            Provider::Muse => Some("muse-image-1.0"),
        }
    }

    /// API root, without a trailing slash.
    pub fn api_base(self) -> &'static str {
        match self {
            Provider::OpenAi => "https://api.openai.com/v1",
            Provider::Gemini => "https://generativelanguage.googleapis.com/v1beta",
            Provider::Muse => "https://api.meta.ai/v1",
        }
    }

    /// Where the user creates a key (for messages).
    pub fn key_page(self) -> &'static str {
        match self {
            Provider::OpenAi => "platform.openai.com",
            Provider::Gemini => "aistudio.google.com/apikey",
            Provider::Muse => "o painel \"API keys\" da Meta Model API",
        }
    }

    /// Cheap sanity check of a key typed by the user (not a guarantee that it works).
    pub fn check_key(self, key: &str) -> Result<()> {
        let k = key.trim();
        if k.is_empty() {
            return Err(GenFillError::new(format!(
                "Chave {} não encontrada.\n\nCole a chave na barra do Preenchimento Generativo ou defina a variável de ambiente {} e abra o programa de novo.",
                self.label(),
                self.env_var()
            )));
        }
        if self == Provider::OpenAi && !k.starts_with("sk-") {
            return Err(GenFillError::new("O que foi informado não parece uma chave da OpenAI (elas começam com \"sk-\"). Confira se você colou a chave certa."));
        }
        if k.len() < 16 || k.chars().any(char::is_whitespace) {
            return Err(GenFillError::new(format!("O que foi informado não parece uma chave válida de {}. Confira se você colou a chave inteira, sem espaços.", self.label())));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_aliases_parse() {
        for p in Provider::ALL {
            assert_eq!(Provider::parse(p.id()), Some(p));
        }
        assert_eq!(Provider::parse(" Google "), Some(Provider::Gemini));
        assert_eq!(Provider::parse("nada"), None);
        assert_eq!(Provider::default(), Provider::OpenAi);
    }

    #[test]
    fn only_openai_uses_a_mask() {
        assert!(Provider::OpenAi.uses_mask());
        assert!(!Provider::Gemini.uses_mask());
        assert!(!Provider::Muse.uses_mask());
        assert_eq!(Provider::Gemini.model(), Some("gemini-3.1-flash-image"));
        assert_eq!(Provider::OpenAi.model(), None);
    }

    #[test]
    fn key_checks_name_the_variable_and_the_shape() {
        assert!(Provider::Gemini.check_key("").unwrap_err().0.contains("GEMINI_API_KEY"));
        assert!(Provider::OpenAi.check_key("abc").unwrap_err().0.contains("sk-"));
        assert!(Provider::OpenAi.check_key("sk-test-1234567890abcd").is_ok());
        assert!(Provider::Gemini.check_key("AIza1234567890abcdef").is_ok());
        assert!(Provider::Muse.check_key("curta").is_err());
        assert!(Provider::Muse.check_key("com espaco 1234567890").is_err());
    }
}
