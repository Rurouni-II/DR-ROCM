"""
Dr. ROCM — Medical Image Triage Assistant
UI     : Streamlit
Vision : Cloudflare Workers AI, via AI Gateway (OpenAI-compatible chat completions)
Memory : lightweight session history + local JSON log
"""

import base64
import io
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


def html_block(s: str) -> str:
    """Strip leading whitespace from every line so Streamlit's Markdown
    parser doesn't mistake indented HTML for a literal code block."""
    return "\n".join(line.strip() for line in s.strip("\n").splitlines())

# ── Config ───────────────────────────────────────────────────────────────────
# Any Workers AI vision model works here. Llama 3.2 11B Vision is the
# established choice for image reasoning / OCR; Llama 4 Scout is a stronger
# (natively multimodal, MoE) alternative if you want to try it.
VISION_MODELS = {
    "Llama 3.2 11B Vision (recommended)": "@cf/meta/llama-3.2-11b-vision-instruct",
    "Llama 4 Scout 17B (multimodal MoE)": "@cf/meta/llama-4-scout-17b-16e-instruct",
    "Mistral Small 3.1 24B (vision)": "@cf/mistralai/mistral-small-3.1-24b-instruct",
}
DEFAULT_MODEL_LABEL = "Llama 3.2 11B Vision (recommended)"
MAX_IMAGE_DIM = 1568  # keep payloads reasonable; the model doesn't benefit past this
LOG_PATH = Path(__file__).parent / "data" / "triage_log.json"

TRIAGE_COLORS = {
    "normal": "#1B9E5A",
    "monitor": "#D9A404",
    "urgent": "#E8720C",
    "emergency": "#D6293E",
    "not_applicable": "#6B7684",
    "unknown": "#6B7684",
}
TRIAGE_LABELS = {
    "normal": "NORMAL",
    "monitor": "MONITOR",
    "urgent": "URGENT",
    "emergency": "EMERGENCY",
    "not_applicable": "N/A",
    "unknown": "UNKNOWN",
}

TRIAGE_PROMPT = """You are a medical image triage assistant. \
Analyze the provided image and return a concise structured assessment.
Classify the image as one of: xray, normal_photo, prescription, or unknown.
If the image looks like a prescription, extract the visible text exactly.
If the image looks like a medical photo or X-ray, give a conservative \
triage label: normal, monitor, urgent, or emergency.
Use the following format exactly, with no markdown and no commentary outside it:
image_type: <xray|normal_photo|prescription|unknown>
triage_label: <normal|monitor|urgent|emergency|not_applicable>
summary: <one short sentence>
findings: <bullet-style semicolon-separated details>
prescription_text: <exact text or none>
follow_up_questions: <up to 3 questions, comma-separated>
{context_block}
Do not provide a final diagnosis."""


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dr. ROCM — Triage Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0E1B2A;
    --ink-soft: #4A5A6A;
    --paper: #F4F7F9;
    --panel: #FFFFFF;
    --line: #DCE3E8;
    --teal: #0E7C7B;
    --teal-deep: #095453;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
code, .mono { font-family: 'IBM Plex Mono', monospace; }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Header band with ECG signature ─────────────────────────────────────── */
.dr-header {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, var(--teal-deep) 0%, var(--teal) 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 22px;
    color: #EAF6F5;
}
.dr-header h1 {
    font-size: 30px; font-weight: 700; margin: 0 0 4px 0; letter-spacing: -0.01em;
}
.dr-header p { font-size: 14px; margin: 0; color: #C9E9E7; max-width: 560px; }
.dr-header .badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 0.06em; padding: 3px 9px; border-radius: 999px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
    margin-top: 12px; color: #EAF6F5;
}
.ecg-wrap { position: absolute; top: 0; right: 0; height: 100%; width: 46%; opacity: 0.5; }
.ecg-wrap svg { height: 100%; width: 100%; }
.ecg-line {
    fill: none; stroke: #EAF6F5; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
    stroke-dasharray: 620; stroke-dashoffset: 620;
    animation: draw 3.2s ease-in-out infinite;
}
@keyframes draw {
    0% { stroke-dashoffset: 620; }
    55% { stroke-dashoffset: 0; }
    100% { stroke-dashoffset: -620; }
}
@media (prefers-reduced-motion: reduce) { .ecg-line { animation: none; stroke-dashoffset: 0; } }

/* ── Panels ──────────────────────────────────────────────────────────────── */
.dr-panel {
    background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
    padding: 20px 22px; margin-bottom: 16px;
}
.dr-panel h3 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--ink-soft); margin: 0 0 14px 0; font-weight: 600;
}

/* ── Report card, chart-tab styling ─────────────────────────────────────── */
.report-card {
    background: var(--panel); border: 1px solid var(--line); border-left: 6px solid var(--tcolor, #6B7684);
    border-radius: 10px; padding: 22px 24px; margin-bottom: 14px;
}
.report-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.triage-chip {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600;
    letter-spacing: 0.06em; padding: 4px 12px; border-radius: 999px; color: #fff;
    background: var(--tcolor, #6B7684);
}
.report-summary { font-size: 17px; font-weight: 600; margin: 10px 0 2px 0; }
.report-meta { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--ink-soft); }
.field-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--ink-soft); margin: 16px 0 4px 0; font-weight: 600;
}
.field-value { font-size: 14.5px; line-height: 1.55; }
.record-id { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #9AA7B2; margin-top: 14px; }

/* ── History rows ────────────────────────────────────────────────────────── */
.hist-row {
    display: flex; align-items: center; gap: 10px; padding: 8px 4px;
    border-bottom: 1px solid var(--line); font-size: 13px;
}
.hist-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

/* Buttons */
.stButton > button {
    background: var(--teal); color: #fff; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.55em 1.2em;
}
.stButton > button:hover { background: var(--teal-deep); color: #fff; }

.disclaimer {
    font-size: 12.5px; color: var(--ink-soft); border-top: 1px solid var(--line);
    padding-top: 14px; margin-top: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    html_block("""
<div class="dr-header">
    <div class="ecg-wrap">
        <svg viewBox="0 0 500 200" preserveAspectRatio="none">
            <path class="ecg-line" d="M0,100 L90,100 L110,100 L125,60 L140,150 L155,20 L170,100 L190,100
                L260,100 L280,100 L295,60 L310,150 L325,20 L340,100 L360,100 L500,100" />
        </svg>
    </div>
    <h1>🩺 Dr. ROCM</h1>
    <p>Upload an X-ray, clinical photo, or prescription and get a structured triage read —
    image type, conservative severity label, key findings, and follow-up questions.</p>
    <span class="badge">TRIAGE ASSISTANCE ONLY · NOT A DIAGNOSIS</span>
</div>
"""),
    unsafe_allow_html=True,
)


# ── Client / secrets ─────────────────────────────────────────────────────────
def get_setting(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    return val


if "history" not in st.session_state:
    st.session_state.history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []
if "pending_questions" not in st.session_state:
    st.session_state.pending_questions = []
if "draft_answers" not in st.session_state:
    st.session_state.draft_answers = {}
if "open_question" not in st.session_state:
    st.session_state.open_question = None


with st.sidebar:
    st.markdown("### Settings")

    _cf_account_id = get_setting("CLOUDFLARE_ACCOUNT_ID")
    _cf_api_token = get_setting("CLOUDFLARE_API_TOKEN")
    _cf_gateway_id = get_setting("CLOUDFLARE_GATEWAY_ID") or "default"

    if _cf_account_id and _cf_api_token:
        # Credentials are configured server-side (env vars / st.secrets) — never
        # echo them back into a visible/editable widget.
        st.success("Cloudflare credentials loaded from server config ✓")
        with st.expander("Use different credentials for this session"):
            _account_override = st.text_input(
                "Cloudflare account ID",
                value="",
                placeholder="Leave blank to use configured account ID",
            )
            _token_override = st.text_input(
                "Cloudflare API token",
                value="",
                type="password",
                placeholder="Leave blank to use configured token",
            )
            _gateway_override = st.text_input(
                "AI Gateway ID",
                value="",
                placeholder=f"Leave blank to use '{_cf_gateway_id}'",
            )
        account_id_input = _account_override or _cf_account_id
        api_token_input = _token_override or _cf_api_token
        gateway_id_input = _gateway_override or _cf_gateway_id
    else:
        st.warning("No Cloudflare credentials configured on the server.")
        account_id_input = st.text_input(
            "Cloudflare account ID",
            value="",
            help="Found on the right side of any zone/dashboard page, or via `wrangler whoami`.",
        )
        api_token_input = st.text_input(
            "Cloudflare API token",
            value="",
            type="password",
            help="A token with the 'Workers AI' permission. Create one under My Profile → API Tokens.",
        )
        gateway_id_input = st.text_input(
            "AI Gateway ID",
            value="default",
            help="Your gateway's name in the AI Gateway dashboard. 'default' is created automatically.",
        )

    model_label = st.selectbox("Vision model", list(VISION_MODELS.keys()), index=0)
    VISION_MODEL = VISION_MODELS[model_label]
    st.caption(f"Model ID: `{VISION_MODEL}`")

    st.markdown("---")
    st.markdown("### Recent triage history")
    if not st.session_state.history:
        st.caption("No analyses yet this session.")
    else:
        for rec in reversed(st.session_state.history[-15:]):
            color = TRIAGE_COLORS.get(rec["triage_label"], "#6B7684")
            st.markdown(
                html_block(f"""<div class="hist-row">
                    <div class="hist-dot" style="background:{color};"></div>
                    <div><b>{rec['image_type']}</b> · {rec['summary'][:48]}</div>
                </div>"""),
                unsafe_allow_html=True,
            )
    if st.session_state.history and st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_report = None
        st.session_state.conversation = []
        st.session_state.chat_display = []
        st.rerun()


# ── Utilities ────────────────────────────────────────────────────────────────
def resize_for_upload(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = MAX_IMAGE_DIM / max(w, h)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)))
    return image


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def triage_text_to_dict(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    if "follow_up_questions" in out:
        out["follow_up_questions"] = [
            q.strip() for q in out["follow_up_questions"].split(",") if q.strip()
        ]
    return out


def log_record(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if LOG_PATH.exists():
        try:
            existing = json.loads(LOG_PATH.read_text())
        except Exception:
            existing = []
    existing.append(record)
    LOG_PATH.write_text(json.dumps(existing[-500:], indent=2))


def call_gateway(
    messages: list,
    model: str,
    account_id: str,
    api_token: str,
    gateway_id: str,
) -> str:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "cf-aig-gateway-id": gateway_id,
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 700}

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if not resp.ok:
        raise RuntimeError(f"Cloudflare AI Gateway error {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def run_triage(
    image: Image.Image,
    patient_context: str,
    model: str,
    account_id: str,
    api_token: str,
    gateway_id: str,
) -> tuple[str, list]:
    """Returns (raw_report_text, initial_messages) — initial_messages seeds the
    follow-up conversation so the image stays in context for later turns."""
    context_block = f"Patient context: {patient_context}\n" if patient_context.strip() else ""
    prompt = TRIAGE_PROMPT.format(context_block=context_block)
    resized = resize_for_upload(image)
    data_url = image_to_data_url(resized)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    reply = call_gateway(messages, model, account_id, api_token, gateway_id)
    messages.append({"role": "assistant", "content": reply})
    return reply, messages


# ── Main layout ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown('<div class="dr-panel"><h3>Intake</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed"
    )
    if uploaded:
        st.image(uploaded, use_container_width=True)
    context = st.text_area(
        "Patient context (optional)",
        placeholder="e.g. 45-year-old male, chest pain for 2 days …",
        height=90,
    )
    run = st.button("🔍 Run triage analysis", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        html_block("""<div class="dr-panel disclaimer">
        ⚠️ This tool is for <b>triage assistance only</b> and does not constitute a medical
        diagnosis. Always consult a qualified healthcare professional.
        </div>"""),
        unsafe_allow_html=True,
    )

with right:
    result_slot = st.container()

    if run:
        if not uploaded:
            st.warning("Please upload an image first.")
        elif not (account_id_input and api_token_input):
            st.error("Add your Cloudflare account ID and API token in the sidebar to run an analysis.")
        else:
            with st.spinner("Analyzing image…"):
                try:
                    image = Image.open(uploaded)
                    t0 = time.time()
                    raw_text, seed_messages = run_triage(
                        image,
                        context,
                        VISION_MODEL,
                        account_id_input,
                        api_token_input,
                        gateway_id_input or "default",
                    )
                    elapsed = time.time() - t0
                    report = triage_text_to_dict(raw_text)

                    record = {
                        "id": str(uuid.uuid4())[:8],
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "image_type": report.get("image_type", "unknown"),
                        "triage_label": report.get("triage_label", "unknown").lower(),
                        "summary": report.get("summary", "—"),
                        "findings": report.get("findings", "—"),
                        "prescription_text": report.get("prescription_text", "none"),
                        "follow_up_questions": report.get("follow_up_questions", []),
                        "patient_context": context,
                        "elapsed_seconds": round(elapsed, 1),
                    }
                    st.session_state.history.append(record)
                    st.session_state.last_report = record
                    st.session_state.conversation = seed_messages
                    st.session_state.pending_questions = list(record["follow_up_questions"] or [])
                    st.session_state.draft_answers = {}
                    st.session_state.open_question = None
                    st.session_state.chat_display = [
                        {
                            "role": "assistant",
                            "content": (
                                f"**{TRIAGE_LABELS.get(record['triage_label'], 'UNKNOWN')}** · "
                                f"{record['summary']}\n\nAsk me anything else about this image, "
                                f"or answer one of the follow-up questions above."
                            ),
                        }
                    ]
                    try:
                        log_record(record)
                    except Exception:
                        pass  # local log is best-effort; never block the UI on it

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    report = st.session_state.last_report
    with result_slot:
        if not report:
            st.markdown(
                html_block("""<div class="dr-panel" style="text-align:center; color:var(--ink-soft); padding:60px 20px;">
                Results will appear here once you run an analysis.
                </div>"""),
                unsafe_allow_html=True,
            )
        else:
            label = report["triage_label"] if report["triage_label"] in TRIAGE_COLORS else "unknown"
            color = TRIAGE_COLORS[label]
            follow_ups = report.get("follow_up_questions") or []
            follow_up_html = (
                "".join(f"<div class='field-value'>• {q}</div>" for q in follow_ups)
                if follow_ups
                else "<div class='field-value'>—</div>"
            )

            st.markdown(
                html_block(f"""
<div class="report-card" style="--tcolor:{color};">
    <div class="report-top">
        <span class="triage-chip">{TRIAGE_LABELS.get(label, "UNKNOWN")}</span>
        <span class="report-meta">{report['image_type']} · {report['elapsed_seconds']}s</span>
    </div>
    <div class="report-summary">{report['summary']}</div>

    <div class="field-label">Findings</div>
    <div class="field-value">{report['findings']}</div>

    <div class="field-label">Prescription text</div>
    <div class="field-value mono">{report['prescription_text']}</div>

    <div class="field-label">Follow-up questions</div>
    {follow_up_html}

    <div class="record-id">record · {report['id']} · {report['timestamp']}</div>
</div>
"""),
                unsafe_allow_html=True,
            )

# ── Follow-up conversation ──────────────────────────────────────────────────
if st.session_state.last_report and st.session_state.get("conversation"):
    st.markdown("### 💬 Continue the conversation")
    st.caption(
        "Click a question below to answer it. Answer as many as you like, then send "
        "them together — the image stays in context for an updated read."
    )

    for turn in st.session_state.chat_display:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # ── Question buttons ────────────────────────────────────────────────────
    if st.session_state.pending_questions:
        st.markdown("**Follow-up questions**")
        for i, q in enumerate(st.session_state.pending_questions):
            answered = q in st.session_state.draft_answers
            label = f"✅ {q}" if answered else f"❓ {q}"
            if st.button(label, key=f"qbtn_{i}", use_container_width=True):
                st.session_state.open_question = q
                st.rerun()

    # ── Text box for whichever question is open ─────────────────────────────
    if st.session_state.open_question:
        q = st.session_state.open_question
        st.markdown(f"**Answering:** {q}")
        answer_text = st.text_area(
            "Your answer", value=st.session_state.draft_answers.get(q, ""), key="answer_box"
        )
        c1, c2 = st.columns(2)
        if c1.button("💾 Save answer", type="primary", use_container_width=True):
            if answer_text.strip():
                st.session_state.draft_answers[q] = answer_text.strip()
            st.session_state.open_question = None
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            st.session_state.open_question = None
            st.rerun()

    custom_msg = st.text_input("Or ask something else (optional)", key="custom_followup")

    if st.session_state.draft_answers or custom_msg.strip():
        if st.button("📤 Send answers to model", type="primary", use_container_width=True):
            if not (account_id_input and api_token_input):
                st.error("Add your Cloudflare credentials in the sidebar to continue.")
            else:
                parts = [f"Q: {q}\nA: {a}" for q, a in st.session_state.draft_answers.items()]
                if custom_msg.strip():
                    parts.append(custom_msg.strip())
                combined = "\n\n".join(parts)

                st.session_state.conversation.append({"role": "user", "content": combined})
                st.session_state.chat_display.append({"role": "user", "content": combined})
                with st.spinner("Thinking…"):
                    try:
                        # cap history so payload/cost don't grow unbounded
                        messages = st.session_state.conversation[-16:]
                        reply = call_gateway(
                            messages,
                            VISION_MODEL,
                            account_id_input,
                            api_token_input,
                            gateway_id_input or "default",
                        )
                        st.session_state.conversation.append({"role": "assistant", "content": reply})
                        st.session_state.chat_display.append({"role": "assistant", "content": reply})
                    except Exception as exc:
                        st.session_state.chat_display.append(
                            {"role": "assistant", "content": f"⚠️ Error: {exc}"}
                        )

                # answered questions drop off the list; anything unanswered stays for later
                st.session_state.pending_questions = [
                    q for q in st.session_state.pending_questions
                    if q not in st.session_state.draft_answers
                ]
                st.session_state.draft_answers = {}
                st.session_state.open_question = None
                st.rerun()
