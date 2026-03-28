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
You are an elite environmental acoustics analyst and sound intelligence expert.

Your task is to infer the most likely environment from the audio metrics below and explain the inference in a rigorous, evidence-based way.

You MUST:
- predict the environment type with probabilities that sum to 100%
- explain WHY each predicted environment is likely
- explicitly map each conclusion to the metrics that support it
- avoid generic statements
- avoid repeating numbers without interpretation
- be specific, confident, and practical
- clearly separate observations, inference, and recommendations

==================== INPUT METRICS ====================

🔊 Loudness / Energy
- dB: {metrics['db']:.2f}
- RMS: {metrics['rms']:.6f}
- Peak: {metrics['peak']:.6f}
- Crest Factor: {metrics['crest']:.2f}

📊 Statistical Shape
- Variance: {metrics['variance']:.6f}
- Std Dev: {metrics['std']:.6f}
- Kurtosis: {metrics['kurtosis']:.2f}
- Skewness: {metrics['skew']:.2f}

⚡ Temporal Behavior
- Zero Crossing Rate (ZCR): {metrics['zcr']:.4f}

🎼 Frequency Profile
- Spectral Centroid: {metrics['centroid']:.2f}
- Spectral Bandwidth: {metrics['bandwidth']:.2f}
- Low Frequency Energy (20–250 Hz): {metrics['low']:.2f}
- Mid Frequency Energy (250–2000 Hz): {metrics['mid']:.2f}
- High Frequency Energy (2000+ Hz): {metrics['high']:.2f}

🌿 Composite Peace Score: {metrics['peace']:.2f}/100

=====================================================

STRICT RULES:
1. Predict the environment, not just the noise level.
2. Give a probability breakdown across likely environments; the probabilities must sum to 100%.
3. For every predicted environment, explain the exact metric evidence used to infer it.
4. Mention counter-evidence too: explain why alternative environments are less likely.
5. Use real-world environment labels such as:
   - quiet room
   - office
   - home interior
   - traffic road
   - vehicle cabin
   - construction site
   - fan / AC room
   - crowd / public place
   - nature / outdoor calm
   - industrial / machinery area
   - café / restaurant
   - classroom / meeting room
6. Be concrete. Do not say vague things like “some noise is present”.
7. Do not claim certainty unless the evidence is extremely strong.
8. If the metrics suggest ambiguity, say so clearly and explain the ambiguity.
9. Do not mention internal reasoning. Give the final evidence-based explanation only.

==================== REQUIRED OUTPUT FORMAT ====================

## 1. Predicted Environment
Give the most likely environment category and a short one-line summary.

## 2. Probability Breakdown
Provide 3 to 5 likely environments with probabilities that sum to 100%.
Example:
- Traffic road — 45%
- Vehicle cabin — 25%
- Office — 15%
- Fan / AC room — 10%
- Quiet indoor room — 5%

## 3. Why This Prediction Was Made
For each environment in the probability breakdown, explain:
- which metrics support it
- what those metrics mean in real-world sound terms
- why it is more or less likely than the others

You must explicitly connect conclusions to metrics such as:
- dB and RMS for loudness
- Peak and Crest Factor for sudden spikes / impulsive events
- Kurtosis for shock-like events such as honks or slams
- Skewness for asymmetry in sound distribution
- ZCR for sharp or noisy behavior
- Spectral Centroid for brightness / hiss / high-frequency dominance
- Bandwidth for spread of frequencies
- Low / Mid / High energy for likely source type

## 4. Noise Characterization
Explain the sound in practical terms:
- steady or fluctuating
- tonal or random
- sharp or dull
- impulsive or continuous
- low-frequency heavy or high-frequency heavy
- human-voice-like or machinery-like or nature-like

## 5. Human Comfort & Perception
Explain how a typical person would perceive this environment:
- calm / moderate / distracting / stressful / harsh
- likely effect on concentration, comfort, and fatigue

## 6. Risk / Impact Assessment
Classify the environment as:
- Quiet
- Moderate
- Noisy
- Potentially harmful

Also explain likely long-exposure impact:
- focus
- stress
- hearing comfort
- annoyance
- fatigue

## 7. Key Metric-to-Meaning Mapping
Provide a point-by-point table or bullet list that maps:
- metric value → interpretation → what it suggests about the environment

Example style:
- High high-frequency energy → hiss / alarms / speech / traffic texture → suggests traffic, crowd, machinery, or outdoor noise
- High low-frequency energy → engine / AC / fan / vehicle hum → suggests vehicle cabin, traffic, industrial room, or AC room

## 8. Practical Recommendations
Give specific, useful advice based on the inferred environment:
- if it is a room: soundproofing / fan placement / window sealing
- if it is outdoors: expected source and mitigation
- if it is a vehicle: cabin noise expectations
- if it is a workplace: acoustic comfort improvements

## 9. Final Verdict
End with a strong 2–3 sentence conclusion that states:
- the most likely environment
- the confidence level
- the main evidence behind the prediction

STYLE REQUIREMENTS:
- Use clear headings
- Use concise but detailed explanations
- Be precise and helpful
- Make it feel like a premium AI acoustic report
- Avoid boilerplate language
"""
    prompt = f"""
Context: This is an environment sound analysis app. The goal is to predict the environment category from audio features and explain the prediction with metric-based evidence.

{prompt}
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
