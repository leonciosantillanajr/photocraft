//! Google Nano Banana 2 (`generateContent`): request body, response parsing and friendly errors.
//!
//! The model takes no mask: the first image already has the area painted with a marker colour.

use base64::Engine as _;
use serde_json::{Value, json};

use crate::openai::EditRequest;
use crate::{GenFillError, Result};

/// Aspect ratios the service accepts, as (w, h) with their text form.
const RATIOS: [(u32, u32, &str); 10] = [
    (1, 1, "1:1"),
    (2, 3, "2:3"),
    (3, 2, "3:2"),
    (3, 4, "3:4"),
    (4, 3, "4:3"),
    (4, 5, "4:5"),
    (5, 4, "5:4"),
    (9, 16, "9:16"),
    (16, 9, "16:9"),
    (21, 9, "21:9"),
];

/// The accepted ratio closest to `w:h`.
pub fn aspect_ratio(w: u32, h: u32) -> &'static str {
    let target = (w.max(1) as f64 / h.max(1) as f64).ln();
    RATIOS
        .iter()
        .min_by(|a, b| {
            let ea = ((a.0 as f64 / a.1 as f64).ln() - target).abs();
            let eb = ((b.0 as f64 / b.1 as f64).ln() - target).abs();
            ea.total_cmp(&eb)
        })
        .map_or("1:1", |r| r.2)
}

/// `1K` / `2K` / `4K` for the longest side we send.
pub fn image_size(w: u32, h: u32) -> &'static str {
    match w.max(h) {
        0..=1100 => "1K",
        1101..=2200 => "2K",
        _ => "4K",
    }
}

fn b64(png: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(png)
}

/// The JSON body: the instruction, then every image (marked crop first, references after).
pub fn request_body(req: &EditRequest) -> Value {
    let mut parts = vec![json!({"text": req.prompt})];
    for img in &req.images_png {
        parts.push(json!({"inlineData": {"mimeType": "image/png", "data": b64(img)}}));
    }
    json!({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio(req.size.0, req.size.1), "imageSize": image_size(req.size.0, req.size.1)},
        },
    })
}

/// Extracts the image from the response (the last image part wins: earlier ones can be drafts).
pub fn parse_response(payload: &Value) -> Result<Vec<u8>> {
    let mut found: Option<&str> = None;
    for cand in payload.get("candidates").and_then(Value::as_array).into_iter().flatten() {
        for part in cand.pointer("/content/parts").and_then(Value::as_array).into_iter().flatten() {
            let data = part.get("inlineData").or_else(|| part.get("inline_data")).and_then(|d| d.get("data")).and_then(Value::as_str);
            if let Some(d) = data {
                found = Some(d);
            }
        }
        if found.is_some() {
            break;
        }
    }
    if let Some(d) = found {
        return base64::engine::general_purpose::STANDARD
            .decode(d.trim())
            .map_err(|e| GenFillError::new(format!("O Google devolveu um arquivo que não é imagem válida: {e}")));
    }
    let block = payload.pointer("/promptFeedback/blockReason").and_then(Value::as_str).unwrap_or("");
    let finish = payload.pointer("/candidates/0/finishReason").and_then(Value::as_str).unwrap_or("");
    let said = payload.pointer("/candidates/0/content/parts/0/text").and_then(Value::as_str).unwrap_or("");
    if !block.is_empty() || finish.contains("SAFETY") || finish.contains("PROHIBITED") || finish.contains("BLOCK") {
        return Err(GenFillError::new("O filtro de segurança do Google bloqueou este pedido. Tente descrever de outro jeito ou escolher outra área."));
    }
    let mut extra = said.split_whitespace().collect::<Vec<_>>().join(" ");
    extra.truncate(extra.char_indices().nth(300).map_or(extra.len(), |(i, _)| i));
    Err(GenFillError::new(if extra.is_empty() {
        "O Google não devolveu nenhuma imagem. Tente de novo (às vezes o pedido é recusado sem explicação).".to_string()
    } else {
        format!("O Google não devolveu nenhuma imagem.\n\nResposta: {extra}")
    }))
}

/// A message in Portuguese for an HTTP error from the Google API.
pub fn friendly_error(status: u16, detail: &str, code: &str, model: &str) -> String {
    let d = detail.to_lowercase();
    let hint = if (status == 400 && (d.contains("api key") || d.contains("api_key"))) || status == 401 {
        "Chave recusada. Confira a chave do Google (Gemini) na barra; você cria uma em aistudio.google.com/apikey.".to_string()
    } else if status == 403 || code == "PERMISSION_DENIED" {
        "Sem permissão. A chave pode não ter acesso a este modelo de imagem, ou o faturamento do projeto Google não está ativado (aistudio.google.com).".to_string()
    } else if status == 404 {
        format!("Modelo '{model}' não encontrado para a sua conta.")
    } else if status == 429 {
        "Limite de uso ou de crédito do Google atingido. Espere um minuto, ou ative o faturamento do projeto na conta Google.".to_string()
    } else if status == 400 {
        "Pedido recusado. Se a mensagem fala de imagem ou tamanho, envie a mensagem abaixo (sem a chave).".to_string()
    } else {
        "Erro do servidor do Google. Tente de novo em instantes.".to_string()
    };
    format!("O Google respondeu com erro {status}.\n{hint}\n\nMensagem original: {}", crate::openai::short(detail, 300))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;

    fn req() -> EditRequest {
        EditRequest {
            provider: Provider::Gemini,
            model: "gemini-3.1-flash-image".into(),
            quality: "medium".into(),
            size: (1536, 1024),
            prompt: "p".into(),
            images_png: vec![vec![1, 2, 3], vec![4]],
            mask_png: Vec::new(),
            moderation: None,
            api_base: "https://x".into(),
            timeout_seconds: 5,
        }
    }

    #[test]
    fn body_has_text_then_images_and_image_config() {
        let b = request_body(&req());
        let parts = b["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0]["text"], "p");
        assert_eq!(parts[1]["inlineData"]["data"], "AQID");
        assert_eq!(b["generationConfig"]["imageConfig"]["aspectRatio"], "3:2");
        assert_eq!(b["generationConfig"]["imageConfig"]["imageSize"], "2K");
        assert!(b.get("tools").is_none());
    }

    #[test]
    fn ratio_and_size_pick_the_closest() {
        assert_eq!(aspect_ratio(1000, 1000), "1:1");
        assert_eq!(aspect_ratio(1920, 1080), "16:9");
        assert_eq!(aspect_ratio(800, 1000), "4:5");
        assert_eq!(aspect_ratio(0, 0), "1:1");
        assert_eq!(image_size(1024, 768), "1K");
        assert_eq!(image_size(2048, 1000), "2K");
        assert_eq!(image_size(3000, 2000), "4K");
    }

    #[test]
    fn parses_the_last_image_part_and_explains_refusals() {
        let ok = json!({"candidates": [{"content": {"parts": [{"text": "aqui"}, {"inlineData": {"mimeType": "image/png", "data": "AQID"}}]}}]});
        assert_eq!(parse_response(&ok).unwrap(), vec![1, 2, 3]);
        let snake = json!({"candidates": [{"content": {"parts": [{"inline_data": {"data": "BA=="}}]}}]});
        assert_eq!(parse_response(&snake).unwrap(), vec![4]);
        let blocked = json!({"promptFeedback": {"blockReason": "SAFETY"}});
        assert!(parse_response(&blocked).unwrap_err().0.contains("segurança"));
        let empty = json!({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "não posso"}]}}]});
        assert!(parse_response(&empty).unwrap_err().0.contains("não posso"));
        assert!(parse_response(&json!({})).unwrap_err().0.contains("nenhuma imagem"));
        assert!(parse_response(&json!({"candidates": [{"content": {"parts": [{"inlineData": {"data": "@@@"}}]}}]})).is_err());
    }

    #[test]
    fn errors_are_friendly() {
        assert!(friendly_error(400, "API key not valid", "", "m").contains("Chave recusada"));
        assert!(friendly_error(403, "x", "PERMISSION_DENIED", "m").contains("Sem permissão"));
        assert!(friendly_error(404, "x", "", "gemini-x").contains("gemini-x"));
        assert!(friendly_error(429, "x", "", "m").contains("Limite"));
        assert!(friendly_error(500, "x", "", "m").contains("servidor do Google"));
    }
}
