import streamlit as st
import numpy as np
import tempfile
import soundfile as sf
from huggingface_hub import InferenceClient
import requests
import os
import io

# ================= CONFIG =================
API_KEY = st.secrets["API_KEY"]  # set this in env
ASR_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

client = InferenceClient(api_key=API_KEY)

# ================= SAVE FLAC =================
def save_flac(uploaded_audio):
    uploaded_audio.seek(0)
    data, samplerate = sf.read(io.BytesIO(uploaded_audio.read()))

    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
        path = f.name

    sf.write(path, data, samplerate, format="FLAC", subtype="PCM_16")
    return path

# ================= TRANSCRIBE =================
def transcribe(flac_path):
    url = f"https://router.huggingface.co/hf-inference/models/{ASR_MODEL}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "audio/flac",
    }

    with open(flac_path, "rb") as f:
        data = f.read()

    for _ in range(3):
        try:
            r = requests.post(url, headers=headers, data=data, timeout=60)
            r.raise_for_status()
            result = r.json()
            return result.get("text", "")
        except Exception as e:
            last_error = e

    raise Exception(f"Transcription failed: {last_error}")

# ================= LLM PROMPT =================
SYSTEM_PROMPT = """
You are an elite AI voice analyst.

Give structured output:

🎯 Key Highlights
🧠 Intent
📌 Action Items
💬 Tone
⚡ Smart Summary
🚀 Improved Version

Keep it sharp, engaging, and insightful.
"""

def analyze(text):
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=600,
        temperature=0.6,
    )
    return completion.choices[0].message.content

# ================= ADVANCED NOISE ANALYSIS =================
def analyze_noise(uploaded_audio):
    uploaded_audio.seek(0)
    data, samplerate = sf.read(io.BytesIO(uploaded_audio.read()))

    if len(data.shape) > 1:
        data = data[:, 0]

    rms = np.sqrt(np.mean(data**2))
    db = 20 * np.log10(rms + 1e-6)

    variance = np.var(data)
    zcr = np.mean(np.abs(np.diff(np.sign(data))))

    fft = np.abs(np.fft.fft(data))
    freqs = np.fft.fftfreq(len(fft), 1/samplerate)

    low = np.mean(fft[(freqs >= 20) & (freqs < 250)])
    mid = np.mean(fft[(freqs >= 250) & (freqs < 2000)])
    high = np.mean(fft[(freqs >= 2000)])

    loudness_score = max(0, 100 - abs(db))
    stability_score = max(0, 100 - variance * 1000)
    sharpness_score = max(0, 100 - zcr * 100)
    freq_score = max(0, 100 - high * 50)

    peace_score = (
        0.3 * loudness_score +
        0.25 * stability_score +
        0.2 * sharpness_score +
        0.25 * freq_score
    )

    return {
        "peace": max(0, min(100, peace_score)),
        "db": db,
        "variance": variance,
        "zcr": zcr,
        "low": low,
        "mid": mid,
        "high": high
    }

# ================= AI NOISE REPORT =================
def noise_report(metrics):
    prompt = f"""
Analyze environment:

dB: {metrics['db']:.2f}
variance: {metrics['variance']:.5f}
zcr: {metrics['zcr']:.4f}
low: {metrics['low']:.2f}
mid: {metrics['mid']:.2f}
high: {metrics['high']:.2f}
peace score: {metrics['peace']:.2f}

Give:
- Environment type
- Noise behavior
- Interpretation
- Suggestions
- Final verdict
"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return completion.choices[0].message.content

# ================= UI =================
st.set_page_config(layout="wide")

st.title("🎙 AI Voice Analyzer + 🌿 Smart Noise Intelligence")

col1, col2 = st.columns(2)

# ===== VOICE ANALYZER =====
with col1:
    st.header("🎙 Voice Analyzer")

    audio = st.audio_input("Record voice")

    if audio:
        st.audio(audio)

        audio.seek(0)
        flac = save_flac(audio)

        with st.spinner("Transcribing..."):
            text = transcribe(flac)

        st.subheader("Transcript")
        st.write(text)

        if text:
            with st.spinner("Analyzing..."):
                result = analyze(text)

            st.subheader("Analysis")
            st.write(result)

# ===== NOISE INTELLIGENCE =====
with col2:
    st.header("🌿 Environment Intelligence")

    env = st.audio_input("Record environment", key="env")

    if env:
        st.audio(env)

        metrics = analyze_noise(env)

        st.metric("Peace Score", f"{metrics['peace']:.2f}")
        st.metric("dB", f"{metrics['db']:.2f}")

        if metrics["peace"] > 70:
            st.success("Peaceful")
        elif metrics["peace"] > 40:
            st.warning("Moderate")
        else:
            st.error("Noisy")

        with st.spinner("Generating AI report..."):
            report = noise_report(metrics)

        st.subheader("AI Noise Report")
        st.write(report)

st.markdown("---")
st.caption("Whisper + HuggingFace + Streamlit 🚀")
