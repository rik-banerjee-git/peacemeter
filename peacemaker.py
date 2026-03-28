import streamlit as st
import numpy as np
import tempfile
import soundfile as sf
from huggingface_hub import InferenceClient
import requests
import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import spectrogram, welch
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import datetime

# ================= CONFIG =================
API_KEY   = st.secrets["API_KEY"]
ASR_MODEL = "openai/whisper-large-v3-turbo"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
client    = InferenceClient(api_key=API_KEY)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SoundLens AI",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #050810;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0e1a !important;
    border-right: 1px solid #1a2040;
}
[data-testid="stSidebar"] * { color: #c8cedd !important; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #0d1228 0%, #111830 50%, #0a1020 100%);
    border: 1px solid #1e2d5a;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(64,120,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
    background: linear-gradient(90deg, #5b8fff, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    color: #6b7fa3;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    margin: 6px 0 0 0;
    letter-spacing: 0.5px;
}

/* Mode Toggle Tabs */
.mode-tab {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.3px;
}

/* Metric Cards */
.metric-card {
    background: #0d1228;
    border: 1px solid #1a2545;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #3a5aad; }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4a5a80;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
}
.metric-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4a5a80;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3a5aad;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1a2545, transparent);
}

/* Transcript box */
.transcript-box {
    background: #0a0f1e;
    border: 1px solid #1a2545;
    border-left: 3px solid #5b8fff;
    border-radius: 10px;
    padding: 18px 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: #c8d8ff;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* Report box */
.report-box {
    background: #070c1a;
    border: 1px solid #1a2545;
    border-radius: 12px;
    padding: 24px 28px;
    font-size: 0.88rem;
    color: #b8c8e8;
    line-height: 1.8;
    white-space: pre-wrap;
    font-family: 'JetBrains Mono', monospace;
}

/* Mic card */
.mic-card {
    background: linear-gradient(135deg, #0d1830, #0a1228);
    border: 1px solid #1e3060;
    border-radius: 14px;
    padding: 20px 24px;
}
.mic-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #7aa2ff;
}
.mic-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4a6090;
    margin-top: 4px;
}

/* Peace gauge */
.peace-bar-wrap {
    background: #0a0f1e;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.peace-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* StStreamlit overrides */
div[data-testid="stMetricValue"] { color: #7aa2ff !important; font-family: 'Syne', sans-serif !important; }
div[data-testid="stMetricLabel"] { color: #4a5a80 !important; }
.stButton > button {
    background: linear-gradient(135deg, #1a3080, #2a48a0) !important;
    color: #e0e8ff !important;
    border: 1px solid #2a4090 !important;
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a4090, #3a58b8) !important;
    border-color: #5b8fff !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #0d2a18, #133820) !important;
    color: #5dde9a !important;
    border: 1px solid #1d5030 !important;
    border-radius: 9px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
div[data-testid="stAudioInput"] > div {
    background: #0a0f1e !important;
    border: 1px dashed #1e3060 !important;
    border-radius: 12px !important;
}
.stSpinner > div { border-top-color: #5b8fff !important; }
[data-testid="stMarkdownContainer"] p { color: #8898b8; }
</style>
""", unsafe_allow_html=True)

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
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "audio/flac"}
    with open(flac_path, "rb") as f:
        data = f.read()
    r = requests.post(url, headers=headers, data=data, timeout=60)
    r.raise_for_status()
    return r.json().get("text", "")

# ================= ENHANCED NOISE ANALYSIS =================
# CHANGES vs original:
#  1. FFT computed only on positive freqs (fixes centroid sign error)
#  2. Added Welch PSD for smoother spectral estimates
#  3. Added LUFS-approximation (loudness standard)
#  4. Added SNR estimate via silence-vs-speech frame split
#  5. Added spectral flatness (tonality indicator)
#  6. Added spectral rolloff (85% energy threshold)
#  7. Added sub-bass (20-80 Hz) & presence (2k-6k Hz) bands
#  8. Added dynamic range metric (peak-to-noise floor)
#  9. Kurtosis/skew now use scipy for numeric stability
# 10. Peace score rebalanced with new metrics

def advanced_noise_analysis(data, sr):
    if len(data.shape) > 1:
        data = data[:, 0]

    # — Core —
    rms   = np.sqrt(np.mean(data**2))
    peak  = np.max(np.abs(data))
    crest = peak / (rms + 1e-9)
    db    = 20 * np.log10(rms + 1e-9)

    # — LUFS approximation (A-weighted RMS in dBFS) —
    lufs = db - 3.0   # simplified; proper LUFS needs filter chain

    # — Statistical (scipy for stability) —
    variance = float(np.var(data))
    std      = float(np.std(data))
    kurt     = float(scipy_kurtosis(data, fisher=True))   # excess kurtosis
    skew     = float(scipy_skew(data))

    # — Temporal —
    zcr    = float(np.mean(np.abs(np.diff(np.sign(data)))))
    energy = float(np.mean(data**2))

    # — SNR estimate: split into 20ms frames, tag bottom 10% as noise floor —
    frame_len   = int(0.02 * sr)
    frames      = [data[i:i+frame_len] for i in range(0, len(data)-frame_len, frame_len)]
    frame_rms   = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    noise_floor = np.percentile(frame_rms, 10)
    snr         = float(20 * np.log10((rms + 1e-9) / (noise_floor + 1e-9)))

    # — Dynamic range —
    dynamic_range = float(20 * np.log10((peak + 1e-9) / (noise_floor + 1e-9)))

    # — Welch PSD —
    nperseg  = min(4096, len(data))
    f_welch, psd = welch(data, sr, nperseg=nperseg)

    # — FFT (positive half) —
    fft_full  = np.fft.fft(data)
    fft_mag   = np.abs(fft_full)
    freqs_all = np.fft.fftfreq(len(fft_full), 1/sr)
    pos_mask  = freqs_all > 0
    fft       = fft_mag[pos_mask]
    freqs     = freqs_all[pos_mask]

    # — Spectral features —
    total_energy  = np.sum(fft) + 1e-9
    centroid      = float(np.sum(freqs * fft) / total_energy)
    bandwidth     = float(np.sqrt(np.sum(((freqs - centroid)**2) * fft) / total_energy))

    # Rolloff: frequency below which 85% of energy lies
    cumulative = np.cumsum(fft)
    rolloff_idx  = np.searchsorted(cumulative, 0.85 * cumulative[-1])
    rolloff       = float(freqs[rolloff_idx]) if rolloff_idx < len(freqs) else float(freqs[-1])

    # Spectral flatness (0=tonal, 1=white noise)
    geometric_mean = np.exp(np.mean(np.log(fft + 1e-9)))
    arithmetic_mean = np.mean(fft)
    flatness        = float(geometric_mean / (arithmetic_mean + 1e-9))

    # Band energies
    sub_bass = float(np.mean(fft[(freqs >= 20)   & (freqs < 80)]))
    low      = float(np.mean(fft[(freqs >= 80)   & (freqs < 250)]))
    mid      = float(np.mean(fft[(freqs >= 250)  & (freqs < 2000)]))
    presence = float(np.mean(fft[(freqs >= 2000) & (freqs < 6000)]))
    high     = float(np.mean(fft[(freqs >= 6000)]))

    # — Enhanced Peace Score —
    # Penalise: loud dB, high variance, high ZCR, high presence/high freq energy, low SNR
    db_norm      = np.clip((db + 60) / 60, 0, 1)     # map -60..0 dB → 0..1
    snr_bonus    = np.clip(snr / 40, 0, 1) * 10      # reward clean signal
    peace = 100 - (
        db_norm   * 25 +
        variance  * 80  * 0.15 +
        zcr       * 40  * 0.15 +
        presence  * 15  * 0.20 +
        high      * 12  * 0.15 +
        (1 - flatness) * 10   # tonal noise is more irritating
    ) + snr_bonus

    return {
        "peace": float(np.clip(peace, 0, 100)),
        "db": float(db), "lufs": float(lufs),
        "rms": float(rms), "peak": float(peak), "crest": float(crest),
        "variance": variance, "std": std, "kurtosis": kurt, "skew": skew,
        "zcr": float(zcr), "snr": float(snr), "dynamic_range": float(dynamic_range),
        "centroid": float(centroid), "bandwidth": float(bandwidth),
        "rolloff": float(rolloff), "flatness": float(flatness),
        "sub_bass": sub_bass, "low": low, "mid": mid,
        "presence": presence, "high": high,
        "data": data, "sr": sr,
        "fft": fft, "freqs": freqs,
        "f_welch": f_welch, "psd": psd,
    }

# ================= VISUALS (Plotly dark) =================
DARK_BG   = "#050810"
GRID_COLOR = "#111830"
ACCENT1   = "#5b8fff"
ACCENT2   = "#a78bfa"
ACCENT3   = "#38bdf8"
ACCENT4   = "#fb923c"

def plotly_base(title, height=300):
    return dict(
        title=dict(text=title, font=dict(family="Syne", size=13, color="#7a9adf")),
        paper_bgcolor=DARK_BG, plot_bgcolor="#07091a",
        font=dict(family="JetBrains Mono", size=10, color="#5a7aaa"),
        height=height,
        margin=dict(l=44, r=16, t=44, b=36),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showgrid=True),
    )

def plot_waveform(data, sr):
    t = np.linspace(0, len(data)/sr, min(len(data), 8000))
    d = data[:len(t)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=d, mode='lines', line=dict(color=ACCENT1, width=0.8), name='Amplitude'))
    layout = plotly_base("🌊 Waveform")
    layout["xaxis"]["title"] = "Time (s)"
    layout["yaxis"]["title"] = "Amplitude"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

def plot_fft(freqs, fft):
    mask = freqs <= 8000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs[mask], y=fft[mask], mode='lines',
                             line=dict(color=ACCENT4, width=0.9), fill='tozeroy',
                             fillcolor='rgba(251,146,60,0.08)', name='Magnitude'))
    layout = plotly_base("📊 FFT Spectrum")
    layout["xaxis"]["title"] = "Frequency (Hz)"
    layout["yaxis"]["title"] = "Magnitude"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

def plot_welch(f_welch, psd):
    mask = f_welch <= 8000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_welch[mask], y=10*np.log10(psd[mask]+1e-12),
                             mode='lines', line=dict(color=ACCENT2, width=1.2),
                             fill='tozeroy', fillcolor='rgba(167,139,250,0.07)', name='PSD'))
    layout = plotly_base("📈 Welch PSD")
    layout["xaxis"]["title"] = "Frequency (Hz)"
    layout["yaxis"]["title"] = "Power (dB/Hz)"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

def plot_spectrogram(data, sr):
    f, t, Sxx = spectrogram(data, sr)
    fig = px.imshow(
        10 * np.log10(Sxx + 1e-10),
        aspect='auto', origin='lower',
        labels=dict(x="Time (s)", y="Frequency (Hz)", color="dB"),
        color_continuous_scale="plasma",
        x=np.round(t, 2), y=np.round(f, 1)
    )
    layout = plotly_base("🔥 Spectrogram", height=350)
    layout["xaxis"]["title"] = "Time (s)"
    layout["yaxis"]["title"] = "Frequency (Hz)"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

def plot_band_bars(metrics):
    bands  = ["Sub-Bass\n20-80Hz", "Low\n80-250Hz", "Mid\n250-2kHz", "Presence\n2-6kHz", "High\n6kHz+"]
    vals   = [metrics["sub_bass"], metrics["low"], metrics["mid"], metrics["presence"], metrics["high"]]
    colors_bar = [ACCENT3, ACCENT1, ACCENT2, ACCENT4, "#f472b6"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bands, y=vals, marker_color=colors_bar, name="Band Energy"))
    layout = plotly_base("🎼 Band Energy Distribution", height=280)
    layout["yaxis"]["title"] = "Energy"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

def plot_peace_gauge(score):
    color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(family="Syne", size=32, color=color), suffix="/100"),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(family="JetBrains Mono", size=9, color="#4a6090")),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#07091a",
            borderwidth=1, bordercolor="#1a2545",
            steps=[
                dict(range=[0, 40],  color="#1a0808"),
                dict(range=[40, 70], color="#0f1208"),
                dict(range=[70, 100],color="#081408"),
            ],
            threshold=dict(line=dict(color=color, width=3), value=score)
        ),
        title=dict(text="☮ PEACE SCORE", font=dict(family="Syne", size=12, color="#4a6090"))
    ))
    fig.update_layout(paper_bgcolor=DARK_BG, height=220, margin=dict(l=20,r=20,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ================= MIC INFO =================
def mic_info_card():
    """Return simulated mic details (browser doesn't expose real values to Python)."""
    st.markdown('<div class="section-title">🎤 Microphone Details</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="mic-card">
  <div class="mic-name">🎙 Built-in Microphone (Default Input Device)</div>
  <div class="mic-detail" style="margin-top:12px; color:#7a9adf; font-size:0.8rem;">
    ⚠ Browser/Streamlit does not expose hardware device names via Python.<br>
    The following are standard specifications for typical laptop/desktop built-in microphones:
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:14px;">
    <div class="metric-card">
      <div class="metric-label">Type</div>
      <div class="metric-value" style="font-size:1rem; color:#7aa2ff;">MEMS / Condenser</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Frequency Response</div>
      <div class="metric-value" style="font-size:1rem; color:#a78bfa;">100 Hz – 16 kHz</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Polar Pattern</div>
      <div class="metric-value" style="font-size:1rem; color:#38bdf8;">Omnidirectional</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Typical SNR</div>
      <div class="metric-value" style="font-size:1rem; color:#fb923c;">58 – 65 dB</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sensitivity</div>
      <div class="metric-value" style="font-size:1rem; color:#f472b6;">−42 dBV/Pa</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sample Rate Used</div>
      <div class="metric-value" style="font-size:1rem; color:#5b8fff;">16 / 44.1 kHz</div>
    </div>
  </div>
  <div style="margin-top:16px; background:#0a1020; border:1px solid #1a3050; border-radius:10px; padding:14px 18px;">
    <div class="metric-label" style="margin-bottom:8px;">📡 Estimated Recording Radius</div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; color:#8ab0e8; line-height:1.9;">
      🔴 <b style="color:#ef4444;">Optimal quality:</b>   0.1 – 0.5 m &nbsp;(close speech, podcasting)<br>
      🟡 <b style="color:#f59e0b;">Usable quality:</b>     0.5 – 1.5 m &nbsp;(desk recording, meetings)<br>
      🟢 <b style="color:#22c55e;">Background pickup:</b>  1.5 – 5.0 m &nbsp;(ambient / room noise)<br>
      ⚪ <b style="color:#6b7fa3;">Near-inaudible:</b>     &gt; 5 m &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(very faint / unusable)
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#3a5080; margin-top:8px;">
      * Radius depends on SPL of source and ambient noise floor. 
        Formula: d = d₀ × 10^((L₀−L_min)/20) where d₀=1m, L₀=94dBSPL, L_min=30dBSPL → ~5m theoretical max.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

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
6. Be concrete. Do not say vague things like "some noise is present".
7. Do not claim certainty unless the evidence is extremely strong.
8. If the metrics suggest ambiguity, say so clearly and explain the ambiguity.
9. Do not mention internal reasoning. Give the final evidence-based explanation only.

==================== REQUIRED OUTPUT FORMAT ====================

## 1. Predicted Environment
- Give TOP 3 probable environments
- Include probability % (must sum ~100)
- Each prediction must reference 1–2 metrics

## 2. Why This Prediction Was Made
For each environment in the probability breakdown, explain:
- which metrics support it
- what those metrics mean in real-world sound terms

## 3.Noise Behavior
- Stable vs fluctuating (use variance/std)
- Spikes or smooth (use crest/kurtosis)
- Tonal vs random (use spectral + ZCR)

## 4. Human Comfort & Perception
Explain how a typical person would perceive this environment:
- calm / moderate / distracting / stressful / harsh
- Link to dB + spectral features
- likely effect on concentration, comfort, and fatigue

## 6. Risk / Impact Assessment
Classify the environment as:
- Quiet
- Moderate
- Noisy
- Potentially harmful
- Based on dB + variability

## 7. Irritation Score (1–10)
- Score + reason using metrics

## 8. Key Insights
- 3–5 sharp observations
- Each must reference a metric

Example style:
- High high-frequency energy → hiss / alarms / speech / traffic texture → suggests traffic, crowd, machinery, or outdoor noise
- High low-frequency energy → engine / AC / fan / vehicle hum → suggests vehicle cabin, traffic, industrial room, or AC room

## 9. Practical Recommendations
Give specific, useful advice based on the inferred environment:
- if it is a room: soundproofing / fan placement / window sealing
- if it is outdoors: expected source and mitigation
- if it is a vehicle: cabin noise expectations
- if it is a workplace: acoustic comfort improvements

## 10. Final Verdict
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
        max_tokens=2000,
    )
    return completion.choices[0].message.content

# ================= PDF REPORT =================
def generate_pdf(metrics, report_text, transcript_text=""):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle('Title2', fontName='Helvetica-Bold', fontSize=20,
                                  textColor=colors.HexColor('#3a6adf'), spaceAfter=4,
                                  alignment=TA_CENTER)
    sub_style    = ParagraphStyle('Sub', fontName='Helvetica', fontSize=9,
                                  textColor=colors.HexColor('#6070a0'),
                                  spaceAfter=16, alignment=TA_CENTER)
    heading_style = ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=12,
                                   textColor=colors.HexColor('#2a5adf'),
                                   spaceBefore=14, spaceAfter=6)
    body_style   = ParagraphStyle('Body2', fontName='Helvetica', fontSize=9,
                                  textColor=colors.HexColor('#222840'),
                                  leading=14, spaceAfter=8)

    story = []
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    story.append(Paragraph("SoundLens AI — Acoustic Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {now}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#c0c8e0'), spaceAfter=16))

    # ── Metrics table ──
    story.append(Paragraph("Core Metrics", heading_style))
    tdata = [
        ["Metric", "Value", "Metric", "Value"],
        ["dB Level",       f"{metrics['db']:.2f} dB",
         "Peace Score",    f"{metrics['peace']:.1f}/100"],
        ["RMS",            f"{metrics['rms']:.5f}",
         "Peak",           f"{metrics['peak']:.5f}"],
        ["Crest Factor",   f"{metrics['crest']:.2f}",
         "SNR",            f"{metrics['snr']:.1f} dB"],
        ["ZCR",            f"{metrics['zcr']:.4f}",
         "Dynamic Range",  f"{metrics['dynamic_range']:.1f} dB"],
        ["Spectral Centroid", f"{metrics['centroid']:.0f} Hz",
         "Spectral Rolloff",  f"{metrics['rolloff']:.0f} Hz"],
        ["Kurtosis",       f"{metrics['kurtosis']:.2f}",
         "Skewness",       f"{metrics['skew']:.2f}"],
        ["Flatness",       f"{metrics['flatness']:.4f}",
         "LUFS (approx.)", f"{metrics['lufs']:.1f}"],
    ]
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a2a6a')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8.5),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f4f6ff')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f4f6ff'), colors.HexColor('#eaeef8')]),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#c8d0e8')),
        ('ALIGN',      (1,0), (1,-1), 'RIGHT'),
        ('ALIGN',      (3,0), (3,-1), 'RIGHT'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ])
    t = Table(tdata, colWidths=[4.2*cm, 3*cm, 4.2*cm, 3*cm])
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Band energy table ──
    story.append(Paragraph("Frequency Band Energy", heading_style))
    bdata = [
        ["Sub-Bass (20-80Hz)", "Low (80-250Hz)", "Mid (250-2kHz)",
         "Presence (2-6kHz)", "High (6kHz+)"],
        [f"{metrics['sub_bass']:.3f}", f"{metrics['low']:.3f}",
         f"{metrics['mid']:.3f}", f"{metrics['presence']:.3f}", f"{metrics['high']:.3f}"],
    ]
    bt = Table(bdata, colWidths=[3.2*cm]*5)
    bt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a4a6a')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8.5),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#eef4fa')),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#c0d0e0')),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ]))
    story.append(bt)
    story.append(Spacer(1, 14))

    # ── Save and embed plots ──
    def fig_to_rl_image(fig_func, *args, w=14*cm, h=5*cm):
        fig_func(*args)
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor='#f8f9ff', edgecolor='none')
        plt.close()
        img_buf.seek(0)
        return RLImage(img_buf, width=w, height=h)

    def draw_waveform(data, sr):
        t = np.linspace(0, len(data)/sr, min(len(data), 6000))
        d = data[:len(t)]
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(t, d, color='#2a5acf', linewidth=0.6)
        ax.set_facecolor('#f8f9ff'); fig.patch.set_facecolor('#f8f9ff')
        ax.set_title("Waveform", fontsize=10); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        ax.grid(color='#dde3f0', linewidth=0.4)

    def draw_fft(freqs, fft):
        mask = freqs <= 8000
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.fill_between(freqs[mask], fft[mask], alpha=0.35, color='#f97316')
        ax.plot(freqs[mask], fft[mask], color='#f97316', linewidth=0.7)
        ax.set_facecolor('#f8f9ff'); fig.patch.set_facecolor('#f8f9ff')
        ax.set_title("FFT Spectrum", fontsize=10); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
        ax.grid(color='#dde3f0', linewidth=0.4)

    def draw_spectrogram(data, sr):
        f_s, t_s, Sxx = spectrogram(data, sr)
        fig, ax = plt.subplots(figsize=(11, 3.5))
        pcm = ax.pcolormesh(t_s, f_s, 10*np.log10(Sxx+1e-10), shading='gouraud', cmap='plasma')
        plt.colorbar(pcm, ax=ax, label='dB')
        ax.set_facecolor('#f8f9ff'); fig.patch.set_facecolor('#f8f9ff')
        ax.set_title("Spectrogram", fontsize=10); ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")

    def draw_bands(metrics_d):
        bands  = ["Sub-Bass\n20-80Hz", "Low\n80-250Hz", "Mid\n250-2kHz", "Presence\n2-6kHz", "High\n6kHz+"]
        vals   = [metrics_d["sub_bass"], metrics_d["low"], metrics_d["mid"], metrics_d["presence"], metrics_d["high"]]
        colors_b = ['#06b6d4','#3b82f6','#8b5cf6','#f97316','#ec4899']
        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.bar(bands, vals, color=colors_b, edgecolor='white', linewidth=0.5)
        ax.set_facecolor('#f8f9ff'); fig.patch.set_facecolor('#f8f9ff')
        ax.set_title("Band Energy Distribution", fontsize=10); ax.set_ylabel("Energy")
        ax.grid(axis='y', color='#dde3f0', linewidth=0.4)

    story.append(Paragraph("Audio Visualizations", heading_style))
    story.append(fig_to_rl_image(draw_waveform, metrics["data"], metrics["sr"], w=15.5*cm, h=4.5*cm))
    story.append(Spacer(1, 8))
    story.append(fig_to_rl_image(draw_fft, metrics["freqs"], metrics["fft"], w=15.5*cm, h=4.5*cm))
    story.append(Spacer(1, 8))
    story.append(fig_to_rl_image(draw_spectrogram, metrics["data"], metrics["sr"], w=15.5*cm, h=5*cm))
    story.append(Spacer(1, 8))
    story.append(fig_to_rl_image(draw_bands, metrics, w=15.5*cm, h=4*cm))
    story.append(Spacer(1, 14))

    # ── Transcript ──
    if transcript_text:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#c0c8e0'), spaceAfter=10))
        story.append(Paragraph("Voice Transcript", heading_style))
        story.append(Paragraph(transcript_text, body_style))
        story.append(Spacer(1, 10))

    # ── AI Report ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#c0c8e0'), spaceAfter=10))
    story.append(Paragraph("AI Acoustic Report", heading_style))
    for line in report_text.split('\n'):
        if line.strip().startswith('##'):
            story.append(Paragraph(line.replace('#','').strip(), heading_style))
        elif line.strip():
            story.append(Paragraph(line.strip(), body_style))

    # ── Footer ──
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#c0c8e0'), spaceAfter=6))
    story.append(Paragraph("Generated by SoundLens AI · Next-gen Audio Intelligence Platform",
                            ParagraphStyle('footer', fontName='Helvetica', fontSize=8,
                                           textColor=colors.HexColor('#8898c0'), alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px 0;">
      <div style="font-size:1.3rem; font-weight:800; color:#5b8fff; letter-spacing:-0.5px;">SoundLens AI</div>
      <div style="font-size:0.7rem; color:#3a5080; font-family:'JetBrains Mono',monospace; margin-top:2px;">v2.0 · Advanced Audio Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem; letter-spacing:1.5px; text-transform:uppercase; color:#3a5080; margin-bottom:8px;">Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["🔬 Noise Profile", "🎙 Audio Analysis"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem; letter-spacing:1.5px; text-transform:uppercase; color:#3a5080; margin-bottom:8px;">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#4a6090; line-height:1.7;">
    • Whisper large-v3-turbo ASR<br>
    • Mistral-7B-Instruct LLM<br>
    • Enhanced 10-metric analysis<br>
    • Welch PSD + spectral rolloff<br>
    • PDF export with graphs
    </div>
    """, unsafe_allow_html=True)

# ================= MAIN HEADER =================
st.markdown("""
<div class="main-header">
  <h1>🎙 SoundLens AI</h1>
  <p>ADVANCED ACOUSTIC INTELLIGENCE PLATFORM · WHISPER ASR · MISTRAL LLM · REAL-TIME ANALYSIS</p>
</div>
""", unsafe_allow_html=True)

# ================= MODE: NOISE PROFILE =================
if mode == "🔬 Noise Profile":
    st.markdown('<div class="section-title">🌿 Environment Noise Profiler</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.82rem; color:#4a5a80; margin-bottom:16px;">Record ambient sound for deep acoustic environment analysis</div>', unsafe_allow_html=True)

    env = st.audio_input("🎤 Record Environment Audio", key="env_noise")

    if env:
        st.audio(env)
        _, data, sr = save_flac(env)
        metrics = advanced_noise_analysis(data, sr)

        # Gauge + top metrics
        col_g, col_m = st.columns([1, 2])
        with col_g:
            plot_peace_gauge(metrics["peace"])
        with col_m:
            st.markdown('<div class="section-title">📊 Key Metrics</div>', unsafe_allow_html=True)
            r1c1, r1c2, r1c3 = st.columns(3)
            r2c1, r2c2, r2c3 = st.columns(3)
            r1c1.metric("🔊 dB Level",     f"{metrics['db']:.1f}")
            r1c2.metric("📶 SNR",           f"{metrics['snr']:.1f} dB")
            r1c3.metric("📈 Crest Factor",  f"{metrics['crest']:.2f}")
            r2c1.metric("⚡ ZCR",           f"{metrics['zcr']:.3f}")
            r2c2.metric("📡 Spectral C.",   f"{metrics['centroid']:.0f} Hz")
            r2c3.metric("🎯 Rolloff",       f"{metrics['rolloff']:.0f} Hz")

        # Extended metrics expander
        with st.expander("🔬 Full Metric Details"):
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("RMS",             f"{metrics['rms']:.5f}")
            e2.metric("Peak",            f"{metrics['peak']:.5f}")
            e3.metric("Variance",        f"{metrics['variance']:.6f}")
            e4.metric("Std Dev",         f"{metrics['std']:.5f}")
            e5, e6, e7, e8 = st.columns(4)
            e5.metric("Kurtosis",        f"{metrics['kurtosis']:.2f}")
            e6.metric("Skewness",        f"{metrics['skew']:.2f}")
            e7.metric("Flatness",        f"{metrics['flatness']:.4f}")
            e8.metric("Dynamic Range",   f"{metrics['dynamic_range']:.1f} dB")
            e9, e10 = st.columns(2)
            e9.metric("Spectral Bandwidth", f"{metrics['bandwidth']:.0f} Hz")
            e10.metric("LUFS (approx.)",    f"{metrics['lufs']:.1f}")

        # Visualizations
        st.markdown('<div class="section-title">📊 Visualizations</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns(2)
        with vc1:
            plot_waveform(metrics["data"], metrics["sr"])
            plot_fft(metrics["freqs"], metrics["fft"])
        with vc2:
            plot_welch(metrics["f_welch"], metrics["psd"])
            plot_band_bars(metrics)
        plot_spectrogram(metrics["data"], metrics["sr"])

        # Mic info
        mic_info_card()

        # AI Report
        st.markdown('<div class="section-title">🤖 AI Acoustic Report</div>', unsafe_allow_html=True)
        with st.spinner("Generating expert acoustic report…"):
            try:
                report = noise_report(metrics)
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

                # PDF download
                st.markdown('<div class="section-title">📥 Export</div>', unsafe_allow_html=True)
                pdf_bytes = generate_pdf(metrics, report)
                st.download_button(
                    "📥 Download Full PDF Report (with graphs)",
                    data=pdf_bytes,
                    file_name=f"soundlens_noise_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Report generation error: {e}")

# ================= MODE: AUDIO ANALYSIS =================
elif mode == "🎙 Audio Analysis":
    st.markdown('<div class="section-title">🎙 Voice & Audio Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.82rem; color:#4a5a80; margin-bottom:16px;">Record voice or any audio for transcription, noise analysis and AI insights — all in one</div>', unsafe_allow_html=True)

    audio = st.audio_input("🎤 Record Audio", key="audio_analysis")

    if audio:
        st.audio(audio)
        flac_path, data, sr = save_flac(audio)
        metrics = advanced_noise_analysis(data, sr)

        # — Transcription —
        st.markdown('<div class="section-title">📝 Transcription</div>', unsafe_allow_html=True)
        transcript = ""
        with st.spinner("Transcribing with Whisper large-v3-turbo…"):
            try:
                transcript = transcribe(flac_path)
                os.unlink(flac_path)
            except Exception as e:
                try: os.unlink(flac_path)
                except: pass
                st.error(f"Transcription error: {e}")

        if transcript:
            st.markdown(f'<div class="transcript-box">{transcript}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="transcript-box" style="color:#3a5080;">(No speech detected or transcription failed)</div>', unsafe_allow_html=True)

        # — Noise metrics —
        st.markdown('<div class="section-title">📊 Audio Metrics</div>', unsafe_allow_html=True)
        col_g, col_m = st.columns([1, 2])
        with col_g:
            plot_peace_gauge(metrics["peace"])
        with col_m:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("🔊 dB",  f"{metrics['db']:.1f}")
            mc2.metric("📶 SNR", f"{metrics['snr']:.1f} dB")
            mc3.metric("⚡ ZCR", f"{metrics['zcr']:.3f}")
            mc4, mc5, mc6 = st.columns(3)
            mc4.metric("📈 Crest",      f"{metrics['crest']:.2f}")
            mc5.metric("📡 Centroid",   f"{metrics['centroid']:.0f} Hz")
            mc6.metric("🎯 Rolloff",    f"{metrics['rolloff']:.0f} Hz")

        # — Plots —
        st.markdown('<div class="section-title">📊 Visualizations</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns(2)
        with vc1:
            plot_waveform(metrics["data"], metrics["sr"])
            plot_fft(metrics["freqs"], metrics["fft"])
        with vc2:
            plot_welch(metrics["f_welch"], metrics["psd"])
            plot_band_bars(metrics)
        plot_spectrogram(metrics["data"], metrics["sr"])

        # — Mic info —
        mic_info_card()

        # — AI Noise Report —
        st.markdown('<div class="section-title">🤖 AI Acoustic + Speech Report</div>', unsafe_allow_html=True)
        with st.spinner("Generating AI report…"):
            try:
                report = noise_report(metrics)
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

                # PDF with transcript
                st.markdown('<div class="section-title">📥 Export</div>', unsafe_allow_html=True)
                pdf_bytes = generate_pdf(metrics, report, transcript_text=transcript)
                st.download_button(
                    "📥 Download Full PDF Report (with graphs + transcript)",
                    data=pdf_bytes,
                    file_name=f"soundlens_audio_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Report error: {e}")

st.markdown("---")
st.markdown('<div style="text-align:center; font-family:JetBrains Mono,monospace; font-size:0.7rem; color:#2a3a60;">🚀 SoundLens AI · Next-gen Audio Intelligence Platform</div>', unsafe_allow_html=True)
