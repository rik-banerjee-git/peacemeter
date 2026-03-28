import streamlit as st
import numpy as np
import tempfile
import soundfile as sf
from huggingface_hub import InferenceClient
import requests
import os
import io
import matplotlib.pyplot as plt

# ================= CONFIG =================
API_KEY = st.secrets["API_KEY"]
ASR_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

client = InferenceClient(api_key=API_KEY)

# ================= SAVE FLAC =================
def save_flac(uploaded_audio):
    uploaded_audio.seek(0)
    data, samplerate = sf.read(io.BytesIO(uploaded_audio.read()))

    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
        path = f.name

    sf.write(path, data, samplerate, format="FLAC", subtype="PCM_16")
    return path, data, samplerate

# ================= TRANSCRIBE =================
def transcribe(flac_path):
    url = f"https://router.huggingface.co/hf-inference/models/{ASR_MODEL}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "audio/flac",
    }

    with open(flac_path, "rb") as f:
        data = f.read()

    r = requests.post(url, headers=headers, data=data, timeout=60)
    r.raise_for_status()
    return r.json().get("text", "")

# ================= ADVANCED NOISE =================
def advanced_noise_analysis(data, sr):
    if len(data.shape) > 1:
        data = data[:, 0]

    # Core
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))
    crest = peak / (rms + 1e-6)
    db = 20 * np.log10(rms + 1e-6)

    # Statistical
    variance = np.var(data)
    std = np.std(data)
    kurtosis = np.mean((data - np.mean(data))**4) / (std**4 + 1e-6)
    skew = np.mean((data - np.mean(data))**3) / (std**3 + 1e-6)

    # Temporal
    zcr = np.mean(np.abs(np.diff(np.sign(data))))
    energy = np.mean(data**2)

    # FFT
    fft = np.abs(np.fft.fft(data))
    freqs = np.fft.fftfreq(len(fft), 1/sr)

    centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-6)
    bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * fft) / (np.sum(fft)+1e-6))

    low = np.mean(fft[(freqs >= 20) & (freqs < 250)])
    mid = np.mean(fft[(freqs >= 250) & (freqs < 2000)])
    high = np.mean(fft[(freqs >= 2000)])

    # Peace score (enhanced)
    peace = 100 - (
        abs(db)*0.3 +
        variance*100*0.2 +
        zcr*50*0.2 +
        high*20*0.3
    )

    return {
        "peace": np.clip(peace, 0, 100),
        "db": db,
        "rms": rms,
        "peak": peak,
        "crest": crest,
        "variance": variance,
        "std": std,
        "kurtosis": kurtosis,
        "skew": skew,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "low": low,
        "mid": mid,
        "high": high,
        "data": data,
        "sr": sr
    }

# ================= VISUALS =================
def plot_waveform(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Waveform")
    st.pyplot(fig)

def plot_fft(data):
    fft = np.abs(np.fft.fft(data))
    fig, ax = plt.subplots()
    ax.plot(fft[:len(fft)//2])
    ax.set_title("FFT Spectrum")
    st.pyplot(fig)

def plot_spectrogram(data, sr):
    fig, ax = plt.subplots()
    ax.specgram(data, Fs=sr)
    ax.set_title("Spectrogram")
    st.pyplot(fig)

# ================= AI NOISE REPORT =================
def noise_report(metrics):
    prompt = f"""
You are a world-class acoustic AI expert.

Analyze deeply:

Core:
- dB: {metrics['db']}
- RMS: {metrics['rms']}
- Peak: {metrics['peak']}
- Crest: {metrics['crest']}

Statistical:
- Variance: {metrics['variance']}
- Std: {metrics['std']}
- Kurtosis: {metrics['kurtosis']}
- Skewness: {metrics['skew']}

Temporal:
- ZCR: {metrics['zcr']}

Frequency:
- Centroid: {metrics['centroid']}
- Bandwidth: {metrics['bandwidth']}
- Low/Mid/High: {metrics['low']}, {metrics['mid']}, {metrics['high']}

Peace Score: {metrics['peace']}

Give EXTREMELY DETAILED:

1. Environment Type (home/office/traffic/nature etc.)
2. Noise Source Probability
3. Acoustic Behavior Explanation
4. Human Comfort Analysis
5. Health Risk Level
6. Irritation Level
7. Improvement Suggestions
8. Final Verdict (Professional tone)
"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
    )

    return completion.choices[0].message.content

# ================= UI =================
st.set_page_config(layout="wide")

st.title("🎙 AI Voice + 🌿 Advanced Environment Intelligence")

col1, col2 = st.columns(2)

# ===== VOICE =====
with col1:
    st.header("🎙 Voice Analyzer")
    audio = st.audio_input("Record voice")

    if audio:
        st.audio(audio)
        flac, data, sr = save_flac(audio)

        with st.spinner("Transcribing..."):
            text = transcribe(flac)

        st.write(text)

# ===== NOISE =====
with col2:
    st.header("🌿 Noise Intelligence")
    env = st.audio_input("Record environment", key="env")

    if env:
        st.audio(env)

        _, data, sr = save_flac(env)

        metrics = advanced_noise_analysis(data, sr)

        st.metric("Peace Score", f"{metrics['peace']:.2f}")
        st.metric("dB", f"{metrics['db']:.2f}")

        # Graphs
        plot_waveform(metrics["data"])
        plot_fft(metrics["data"])
        plot_spectrogram(metrics["data"], metrics["sr"])

        with st.spinner("Generating AI Report..."):
            report = noise_report(metrics)

        st.subheader("📊 AI Noise Report")
        st.write(report)

        # SHARE / DOWNLOAD
        report_text = f"""
Peace Score: {metrics['peace']}

Full Report:
{report}
"""
        st.download_button("📥 Download Report", report_text)

st.markdown("---")
st.caption("🚀 Next-gen Audio Intelligence Platform")
