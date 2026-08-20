# Dr. ROCM — Medical Image Triage Assistant

Streamlit app that takes an uploaded image (X-ray, clinical photo, or prescription)
and returns a structured triage read using a vision model hosted on **Cloudflare
Workers AI**, called through **Cloudflare AI Gateway** (for logging, caching, and
rate limiting). No model runs locally, so it deploys cleanly on Streamlit Community
Cloud's free tier.

⚠️ Triage assistance only — not a diagnosis.

## Getting Cloudflare credentials

1. **Account ID** — visible on the right sidebar of any page in the
   [Cloudflare dashboard](https://dash.cloudflare.com), or run `wrangler whoami`.
2. **API token** — My Profile → API Tokens → Create Token → grant it the
   **Workers AI: Read** permission (this is required even for third-party/gateway
   calls, per Cloudflare's AI REST API).
3. **Gateway ID** — go to AI → AI Gateway in the dashboard. A gateway named
   `default` is created automatically the first time you use it, or you can create
   a named one for its own logs/analytics.

## Run locally

```bash
pip install -r requirements.txt
export CLOUDFLARE_ACCOUNT_ID=your_account_id
export CLOUDFLARE_API_TOKEN=your_api_token
export CLOUDFLARE_GATEWAY_ID=default
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at
   `app.py` in that repo.
3. In the app's **Settings → Secrets**, add:

   ```toml
   CLOUDFLARE_ACCOUNT_ID = "your_account_id"
   CLOUDFLARE_API_TOKEN = "your_api_token"
   CLOUDFLARE_GATEWAY_ID = "default"
   ```

4. Deploy. (You can also paste credentials directly into the sidebar at runtime
   instead of storing them as secrets — handy for quick demos.)

## Choosing a model

The sidebar lets you switch between three Workers AI vision models without touching
code:

- **Llama 3.2 11B Vision** (default) — Meta's dedicated vision-instruct model, a
  solid general choice for image reasoning and OCR-style extraction (useful for the
  prescription-text case).
- **Llama 4 Scout 17B** — natively multimodal MoE model, generally stronger
  reasoning, a bit more expensive per token.
- **Mistral Small 3.1 24B** — adds vision on top of Mistral's small dense model.

Swap the `VISION_MODELS` dict in `app.py` to add others from Cloudflare's
[Workers AI model catalog](https://developers.cloudflare.com/workers-ai/models/)
(anything tagged "Vision").

## Notes on this version

- **No local model.** The original build loaded Qwen2-VL-2B locally via
  `transformers`/`torch` on a Hugging Face ZeroGPU Space. That doesn't run on
  Streamlit Cloud (no GPU, ~1GB RAM), so image analysis now calls a hosted model
  through Cloudflare AI Gateway instead.
- **No ChromaDB / sentence-transformers.** Vector memory added a second heavy
  `torch`-based dependency and wrote to local disk, which Streamlit Cloud wipes on
  every redeploy anyway. Triage history is now kept in the session (sidebar) and
  best-effort logged to `data/triage_log.json` on the running instance — treat that
  file as ephemeral, not a database.
- If you outgrow the free tier's storage limits and want real persistent memory,
  swap `log_record()` for a call to a hosted store (Postgres, Supabase, a hosted
  vector DB, etc.) — the rest of the app doesn't need to change.
