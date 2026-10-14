# Preenchimento Generativo (Generative Fill)

Camada nova gerada pela OpenAI dentro da seleção, como o Preenchimento Generativo do Photoshop.
Só funciona no aplicativo de computador (a versão web não tem acesso à rede).

## Como usar

1. Faça uma seleção (marquee, laço, varinha…). Uma barra flutuante aparece sobre a imagem.
2. **Preenchimento generativo**: descreva o que criar (ou deixe vazio para continuar a cena) e
   clique em **Gerar** (ou Enter). O botão de imagem anexa até 4 referências.
3. **Remover**: apaga o que está selecionado e reconstrói o fundo.
4. **Preencher áreas vazias**: preenche as áreas transparentes da imagem, uma camada por área.
5. **Cancelar seleção**: remove a seleção.

O resultado entra como **camada nova** (nome "Generative Fill") com borda suavizada, em um único
passo de desfazer (Ctrl+Z). Arraste a barra pela alça de pontinhos; a posição é lembrada.

## Chave da OpenAI

- Variável de ambiente `OPENAI_API_KEY` (tem prioridade), **ou**
- cole a chave (`sk-…`) na própria barra quando ela pedir; fica nas configurações do programa.

O saldo da API (platform.openai.com › Billing) é separado da assinatura do ChatGPT. Os modelos de
imagem podem exigir a "Verificação da Organização" na OpenAI.

## Como funciona

`crates/genfill` (lógica pura, sem dependência do restante do projeto) planeja o recorte de
contexto ao redor da seleção (margem 50%, mínimo 512 px, proporção aceita pela API), escolhe a
qualidade pelo tamanho do recorte (baixa ≤ 1024 px, média ≤ 2048 px, alta acima), monta máscara
e prompt e chama `POST /v1/images/edits` (corpo JSON, `ureq` + rustls). O motor
(`crates/engine/src/genfill_cmds.rs`) expõe os comandos abaixo, que rodam como tarefa em segundo
plano cancelável. A interface (`crates/ui-egui/src/genfill_bar.rs`) só desenha a barra.

| Comando | Parâmetros |
|---|---|
| `edit.generativeFill` | `prompt`, `references: [caminhos]`, `quality: auto\|baixa\|media\|alta` |
| `edit.generativeRemove` | `quality` |
| `edit.generativeFillEmpty` | `prompt`, `references`, `quality`, `maxRegions` (1–16, padrão 8) |

Os testes usam uma API falsa (`Backend`): nenhum teste chama a OpenAI nem gasta crédito. O
`panic_hunt` desliga a rede (`genfill_cmds::disable_network`).

## Limites conhecidos

- Cancelar interrompe a tarefa no programa, mas o pedido já enviado à OpenAI pode ser cobrado.
- Só o formato de pedido JSON (`multipart` fica para depois).
- O pacote do Windows é x86_64 (não há build específico para Windows ARM64).
