//! OpenAI image edits (`POST /images/edits`, JSON body): request building, response parsing,
//! friendly error messages, and the [`Backend`] seam that tests replace with a fake.

use base64::Engine as _;
use serde_json::{Value, json};

use crate::plan::{Config, Tier};
use crate::provider::Provider;
use crate::{GenFillError, Result};

/// Everything one edit call needs, already encoded.
#[derive(Clone, Debug)]
pub struct EditRequest {
    pub provider: Provider,
    pub model: String,
    pub quality: String,
    pub size: (u32, u32),
    pub prompt: String,
    /// PNG bytes: the crop first, then the references.
    pub images_png: Vec<Vec<u8>>,
    /// PNG with alpha: transparent = edit here. Empty for services without a mask (the first image is marked instead).
    pub mask_png: Vec<u8>,
    pub moderation: Option<String>,
    pub api_base: String,
    pub timeout_seconds: u64,
}

/// Why a call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    /// The server answered with an error status.
    Http { status: u16, detail: String, code: String },
    /// No answer (no internet, timeout, unreadable reply, no key, desktop-only…): the message is final.
    Other(String),
}

/// Sends one edit and returns the PNG (or other image) bytes of the result.
pub trait Backend: Send + Sync {
    fn edit(&self, req: &EditRequest) -> std::result::Result<Vec<u8>, BackendError>;
}

fn data_url(png: &[u8]) -> String {
    format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(png))
}

/// The JSON body of the request.
pub fn request_body(req: &EditRequest) -> Value {
    let mut body = json!({
        "model": req.model,
        "prompt": req.prompt,
        "n": 1,
        "quality": req.quality,
        "output_format": "png",
        "size": format!("{}x{}", req.size.0, req.size.1),
        "images": req.images_png.iter().map(|d| json!({"image_url": data_url(d)})).collect::<Vec<_>>(),
        "mask": {"image_url": data_url(&req.mask_png)},
    });
    if let (Some(m), Some(o)) = (&req.moderation, body.as_object_mut()) {
        o.insert("moderation".into(), json!(m));
    }
    body
}

/// The JSON body for Meta Muse Image (OpenAI-style endpoint, no mask: the first image is marked).
pub fn muse_body(req: &EditRequest) -> Value {
    json!({
        "model": req.model,
        "prompt": req.prompt,
        "n": 1,
        "size": format!("{}x{}", req.size.0, req.size.1),
        "images": req.images_png.iter().map(|d| json!({"image_url": data_url(d)})).collect::<Vec<_>>(),
    })
}

/// A link to the image when the answer carries one instead of the bytes.
pub fn find_image_url(payload: &Value) -> Option<String> {
    payload.get("data").and_then(Value::as_array)?.iter().find_map(|i| i.get("url").and_then(Value::as_str)).map(str::to_string)
}

/// Extracts the image bytes from the response JSON.
pub fn parse_response(payload: &Value) -> Result<Vec<u8>> {
    for item in payload.get("data").and_then(Value::as_array).into_iter().flatten() {
        if let Some(b64) = item.get("b64_json").and_then(Value::as_str) {
            return base64::engine::general_purpose::STANDARD
                .decode(b64.trim())
                .map_err(|e| GenFillError::new(format!("A OpenAI devolveu um arquivo que não é imagem válida: {e}")));
        }
        if item.get("url").and_then(Value::as_str).is_some() {
            return Err(GenFillError::new("A OpenAI devolveu um link em vez da imagem (formato não suportado pelo programa)."));
        }
    }
    let mut shown = payload.to_string();
    shown.truncate(shown.char_indices().nth(400).map_or(shown.len(), |(i, _)| i));
    Err(GenFillError::new(format!("A OpenAI não devolveu nenhuma imagem.\n\nResposta: {shown}")))
}

pub(crate) fn short(text: &str, n: usize) -> String {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() <= n { t } else { format!("{}…", t.chars().take(n).collect::<String>()) }
}

/// A message in Portuguese for an HTTP error.
pub fn friendly_error(status: u16, detail: &str, code: &str, model: &str) -> String {
    let d = detail.to_lowercase();
    let hint = if status == 401 {
        "Chave recusada. Confira a chave da API (ela começa com sk-) na variável OPENAI_API_KEY ou nas Preferências.".to_string()
    } else if status == 403 || d.contains("must be verified") || d.contains("verification") {
        "Sem permissão para usar este modelo. A OpenAI pode exigir a \"Verificação da Organização\" para modelos de imagem: veja em platform.openai.com, Settings > Organization > General.".to_string()
    } else if status == 404 || (d.contains("model") && (d.contains("does not exist") || d.contains("not found"))) {
        format!("Modelo '{model}' não encontrado para a sua conta.")
    } else if status == 429 && (code == "insufficient_quota" || d.contains("quota") || d.contains("billing") || d.contains("credit")) {
        "Sem saldo/crédito na API. Atenção: o saldo da API (platform.openai.com > Billing) é separado da assinatura do ChatGPT.".to_string()
    } else if status == 429 {
        "Muitos pedidos em pouco tempo. Espere um minuto e tente de novo.".to_string()
    } else if code == "moderation_blocked" || d.contains("safety") || d.contains("moderation") {
        "O filtro de segurança da OpenAI bloqueou este pedido. Tente descrever de outro jeito.".to_string()
    } else if status == 400 {
        "Pedido recusado. Se a mensagem fala de máscara, tamanho ou imagem, envie a mensagem abaixo (sem a chave).".to_string()
    } else {
        "Erro do servidor da OpenAI. Tente de novo em instantes.".to_string()
    };
    format!("A OpenAI respondeu com erro {status}.\n{hint}\n\nMensagem original: {}", short(detail, 300))
}

/// A message in Portuguese for an HTTP error from `provider`.
pub fn friendly_error_for(provider: Provider, status: u16, detail: &str, code: &str, model: &str) -> String {
    match provider {
        Provider::OpenAi => friendly_error(status, detail, code, model),
        Provider::Gemini => crate::gemini::friendly_error(status, detail, code, model),
        Provider::Muse => {
            let hint = match status {
                401 | 403 => "Chave recusada. Confira a chave da Meta Model API (você cria no painel \"API keys\"; ela aparece só uma vez).",
                404 => "Endereço ou modelo não encontrado na Meta Model API.",
                429 => "Muitos pedidos ou sem crédito na Meta Model API. Espere um minuto e tente de novo.",
                400 | 422 => "Pedido recusado pela Meta. Envie a mensagem abaixo (sem a chave) para eu ajustar o formato.",
                _ => "Erro do servidor da Meta. Tente de novo em instantes.",
            };
            format!("A Meta respondeu com erro {status}.\n{hint}\n\nMensagem original: {}", short(detail, 300))
        }
    }
}

/// Builds the request for `tier`, sends it through `backend` and returns the image bytes.
///
/// `images_png` has the crop first (already marked when the service takes no mask), then the references.
#[allow(clippy::too_many_arguments)]
pub fn call(backend: &dyn Backend, cfg: &Config, provider: Provider, tier: &Tier, prompt: String, images_png: Vec<Vec<u8>>, mask_png: Vec<u8>, size: (u32, u32)) -> Result<Vec<u8>> {
    let req = EditRequest {
        provider,
        model: provider.model().map_or_else(|| tier.model.clone(), str::to_string),
        quality: tier.quality.to_string(),
        size,
        prompt,
        images_png,
        mask_png,
        moderation: cfg.moderation.clone(),
        api_base: if provider == Provider::OpenAi { cfg.api_base.clone() } else { provider.api_base().to_string() },
        timeout_seconds: cfg.timeout_seconds,
    };
    backend.edit(&req).map_err(|e| match e {
        BackendError::Http { status, detail, code } => GenFillError::new(friendly_error_for(provider, status, &detail, &code, &req.model)),
        BackendError::Other(m) => GenFillError::new(m),
    })
}

/// Real network backend (native builds only): one service, one key.
#[cfg(not(target_arch = "wasm32"))]
pub struct UreqBackend {
    provider: Provider,
    api_key: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Debug for UreqBackend {
    // Never print the key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UreqBackend({:?}, <key hidden>)", self.provider)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl UreqBackend {
    /// The provider's environment variable wins; `fallback` (the value saved in the program) is used when it is empty.
    pub fn from_env_or(provider: Provider, fallback: &str) -> Result<Self> {
        let from_env = provider.env_vars().iter().filter_map(|v| std::env::var(v).ok()).map(|k| k.trim().to_string()).find(|k| !k.is_empty());
        let key = from_env.unwrap_or_else(|| fallback.trim().to_string());
        Self::new(provider, key)
    }

    pub fn new(provider: Provider, api_key: String) -> Result<Self> {
        provider.check_key(&api_key)?;
        Ok(Self { provider, api_key: api_key.trim().to_string() })
    }

    fn agent(timeout_seconds: u64) -> ureq::Agent {
        use std::time::Duration;
        ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(timeout_seconds.max(1)))).http_status_as_error(false).build().into()
    }

    fn read_error(&self, status: u16, json: &Value) -> BackendError {
        let err = json.get("error");
        let detail = err.and_then(|e| e.get("message")).and_then(Value::as_str).unwrap_or("").to_string();
        let code = err.and_then(|e| e.get("code").or_else(|| e.get("status")).or_else(|| e.get("type"))).and_then(Value::as_str).unwrap_or("").to_string();
        BackendError::Http { status, detail, code }
    }

    fn download(agent: &ureq::Agent, url: &str) -> std::result::Result<Vec<u8>, BackendError> {
        if !url.starts_with("https://") {
            return Err(BackendError::Other("A Meta devolveu um link que não é seguro (https); não vou baixar.".into()));
        }
        let mut resp = agent.get(url).call().map_err(|e| BackendError::Other(format!("Não consegui baixar a imagem gerada.\n({e})")))?;
        if resp.status().as_u16() >= 400 {
            return Err(BackendError::Other("Não consegui baixar a imagem gerada (o link recusou o pedido).".into()));
        }
        resp.body_mut().with_config().limit(256 * 1024 * 1024).read_to_vec().map_err(|e| BackendError::Other(format!("Não consegui baixar a imagem gerada.\n({e})")))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Backend for UreqBackend {
    fn edit(&self, req: &EditRequest) -> std::result::Result<Vec<u8>, BackendError> {
        let agent = Self::agent(req.timeout_seconds);
        let base = req.api_base.trim_end_matches('/');
        let (url, body) = match self.provider {
            Provider::OpenAi => (format!("{base}/images/edits"), request_body(req)),
            Provider::Gemini => (format!("{base}/models/{}:generateContent", req.model), crate::gemini::request_body(req)),
            Provider::Muse => (format!("{base}/images/edits"), muse_body(req)),
        };
        let builder = agent.post(&url).header("Content-Type", "application/json");
        let builder = match self.provider {
            Provider::Gemini => builder.header("x-goog-api-key", &self.api_key),
            _ => builder.header("Authorization", &format!("Bearer {}", self.api_key)),
        };
        let mut resp = builder.send(body.to_string().as_bytes()).map_err(|e| match e {
            ureq::Error::Timeout(_) => BackendError::Other("O servidor demorou demais para responder. Tente de novo.".into()),
            other => BackendError::Other(format!("Não consegui falar com o servidor. Verifique a internet.\n({other})")),
        })?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .with_config()
            .limit(256 * 1024 * 1024)
            .read_to_string()
            .map_err(|e| BackendError::Other(format!("Não consegui ler a resposta do servidor.\n({e})")))?;
        let json: Value = serde_json::from_str(&text).map_err(|_| {
            if status >= 400 {
                BackendError::Http { status, detail: short(&text, 300), code: String::new() }
            } else {
                BackendError::Other("O servidor respondeu algo que não consegui entender.".into())
            }
        })?;
        if status >= 400 {
            return Err(self.read_error(status, &json));
        }
        match self.provider {
            Provider::Gemini => crate::gemini::parse_response(&json),
            Provider::OpenAi => parse_response(&json),
            Provider::Muse => match find_image_url(&json) {
                Some(url) if json.pointer("/data/0/b64_json").is_none() => return Self::download(&agent, &url),
                _ => parse_response(&json),
            },
        }
        .map_err(|e| BackendError::Other(e.0))
    }
}

/// On the web build there is no network client.
#[cfg(target_arch = "wasm32")]
pub struct UreqBackend;

#[cfg(target_arch = "wasm32")]
impl UreqBackend {
    pub fn from_env_or(_provider: Provider, _fallback: &str) -> Result<Self> {
        Err(GenFillError::new("O Preenchimento Generativo só funciona no aplicativo de computador, não na versão web."))
    }
}

#[cfg(target_arch = "wasm32")]
impl Backend for UreqBackend {
    fn edit(&self, _req: &EditRequest) -> std::result::Result<Vec<u8>, BackendError> {
        Err(BackendError::Other("O Preenchimento Generativo só funciona no aplicativo de computador, não na versão web.".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> EditRequest {
        EditRequest {
            provider: Provider::OpenAi,
            model: "m".into(),
            quality: "low".into(),
            size: (1024, 768),
            prompt: "p".into(),
            images_png: vec![vec![1, 2, 3], vec![4]],
            mask_png: vec![9],
            moderation: Some("auto".into()),
            api_base: "https://x/v1".into(),
            timeout_seconds: 5,
        }
    }

    #[test]
    fn body_has_images_mask_and_size() {
        let b = request_body(&req());
        assert_eq!(b["size"], "1024x768");
        assert_eq!(b["images"].as_array().unwrap().len(), 2);
        assert_eq!(b["images"][0]["image_url"], "data:image/png;base64,AQID");
        assert_eq!(b["mask"]["image_url"], "data:image/png;base64,CQ==");
        assert_eq!(b["moderation"], "auto");
        let mut r = req();
        r.moderation = None;
        assert!(request_body(&r).get("moderation").is_none());
    }

    #[test]
    fn parses_b64_and_rejects_other_shapes() {
        assert_eq!(parse_response(&json!({"data": [{"b64_json": "AQID"}]})).unwrap(), vec![1, 2, 3]);
        assert!(parse_response(&json!({"data": [{"url": "http://x"}]})).unwrap_err().0.contains("link"));
        assert!(parse_response(&json!({"data": []})).unwrap_err().0.contains("nenhuma imagem"));
        assert!(parse_response(&json!({"data": [{"b64_json": "@@@"}]})).is_err());
    }

    #[test]
    fn errors_are_friendly() {
        assert!(friendly_error(401, "bad", "", "m").contains("Chave recusada"));
        assert!(friendly_error(403, "Your organization must be verified", "", "m").contains("Verificação"));
        assert!(friendly_error(404, "x", "", "gpt-x").contains("gpt-x"));
        assert!(friendly_error(429, "You exceeded your current quota", "insufficient_quota", "m").contains("saldo"));
        assert!(friendly_error(429, "slow down", "", "m").contains("Muitos pedidos"));
        assert!(friendly_error(400, "blocked", "moderation_blocked", "m").contains("filtro de segurança"));
        assert!(friendly_error(500, "oops", "", "m").contains("servidor da OpenAI"));
    }

    #[test]
    fn muse_body_has_no_mask_and_links_are_found() {
        let mut r = req();
        r.provider = Provider::Muse;
        let b = muse_body(&r);
        assert!(b.get("mask").is_none() && b.get("quality").is_none());
        assert_eq!(b["images"].as_array().unwrap().len(), 2);
        assert_eq!(find_image_url(&json!({"data": [{"url": "https://x/i.png"}]})).as_deref(), Some("https://x/i.png"));
        assert_eq!(find_image_url(&json!({"data": []})), None);
        assert!(friendly_error_for(Provider::Muse, 401, "x", "", "m").contains("Chave recusada"));
        assert!(friendly_error_for(Provider::Gemini, 429, "x", "", "m").contains("Google"));
        assert!(friendly_error_for(Provider::OpenAi, 429, "slow", "", "m").contains("Muitos pedidos"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn key_checks() {
        assert!(UreqBackend::new(Provider::OpenAi, String::new()).unwrap_err().0.contains("não encontrada"));
        assert!(UreqBackend::new(Provider::OpenAi, "abc".into()).unwrap_err().0.contains("sk-"));
        assert!(UreqBackend::new(Provider::OpenAi, "sk-test-1234567890abcd".into()).is_ok());
        assert!(UreqBackend::new(Provider::Gemini, "AIza1234567890abcdef".into()).is_ok());
        assert!(UreqBackend::new(Provider::Muse, String::new()).unwrap_err().0.contains("MODEL_API_KEY"));
    }
}
