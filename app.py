import os
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft
from scipy.stats import norm
from transformers import pipeline
import soundfile as sf  # To read audio files without audioread

# Cache the Sentiment Model (Loads only once)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_model()

# Cache the Chartwell Logo Download (Loads only once)
@st.cache_data
def load_chartwell_logo(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

chartwell_logo_path = "https://raw.githubusercontent.com/Yash9808/Call_Analysis_V4one/main/The Chartwell Hospital (Logo).png"
img = load_chartwell_logo(chartwell_logo_path)

# Display Logo
col1, col2, col3 = st.columns([5, 10, 1])  
with col2:  
    st.image(img, width=150)

# Streamlit UI
st.title("🎤 Single Audio-Call Sentiment Analysis")
st.write("Upload an MP3 file to analyze its sentiment.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

# Cache the Audio Analysis Function (Runs once per file)
@st.cache_data
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Compute FFT
    fft_vals = np.abs(fft(y))[:len(y)//2]
    freqs = np.linspace(0, sr/2, len(fft_vals))
    peak_freq = freqs[np.argmax(fft_vals)]

    # Estimate Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    
    if len(pitch_values) > 0:
        pitch_mean, pitch_std = np.mean(pitch_values), np.std(pitch_values)
    else:
        pitch_mean, pitch_std = 0, 0

    # Determine if peak frequency is high or low
    threshold = 300  
    peak_color = 'red' if peak_freq > threshold else 'green'

    # Sentiment analysis
    sentiment_result = sentiment_analyzer("This is a placeholder for sentiment analysis based on audio!")

    return sentiment_result[0], mfccs_mean, freqs, fft_vals, pitch_mean, pitch_std, peak_freq, peak_color

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Run analysis function
    sentiment, mfccs, freqs, fft_vals, pitch_mean, pitch_std, peak_freq, peak_color = analyze_audio(file_path)

    ### **📊 Sentiment Analysis Results**
    st.subheader("📊 Sentiment Analysis Result")
    st.write(f"**Sentiment:** {sentiment['label']}")
    st.write(f"**Confidence:** {sentiment['score']:.2f}")

    ### **🔹 MFCC Analysis**
    st.subheader("🎵 MFCC Feature Extraction")
    st.write("MFCC (Mel-Frequency Cepstral Coefficients) helps analyze the quality and tone of speech.")

    mfcc_quality = 'Good' if np.mean(mfccs) > -100 else 'Bad'
    mfcc_color = 'green' if mfcc_quality == 'Good' else 'red'
    st.markdown(f"**MFCC Quality:** <span style='color:{mfcc_color}'>{mfcc_quality}</span>", unsafe_allow_html=True)

    # MFCC Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(mfccs)), mfccs)
    ax.set_xlabel("MFCC Coefficients")
    ax.set_ylabel("Mean Value")
    ax.set_title("MFCC Feature Extraction")
    st.pyplot(fig)

    ### **🔹 FFT Analysis**
    st.subheader("📈 FFT (Frequency Analysis)")
    st.write("FFT (Fast Fourier Transform) helps analyze the frequency content of the voice.")

    fft_quality = 'High' if peak_freq > 300 else 'Normal'
    fft_color = 'red' if fft_quality == 'High' else 'green'
    st.markdown(f"**FFT Peak Frequency:** <span style='color:{fft_color}'>{peak_freq:.2f} Hz ({fft_quality})</span>", unsafe_allow_html=True)

    # FFT Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, fft_vals, label='FFT Spectrum')
    ax.axvline(peak_freq, color=peak_color, linestyle='--', label=f'Peak: {peak_freq:.2f} Hz')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("FFT of Voice Data")
    ax.legend()
    st.pyplot(fig)

    ### **🔹 Pitch Analysis**
    st.subheader("🎵 Pitch Distribution")
    st.write("Pitch helps identify the general tone of the voice.")

    pitch_quality = 'Good' if 100 < pitch_mean < 300 else 'Bad'
    pitch_color = 'green' if pitch_quality == 'Good' else 'red'
    st.markdown(f"**Pitch Quality:** <span style='color:{pitch_color}'>{pitch_mean:.2f} Hz ({pitch_quality})</span>", unsafe_allow_html=True)

    # Pitch Bell Curve Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    if pitch_std > 0:
        x_vals = np.linspace(pitch_mean - 3*pitch_std, pitch_mean + 3*pitch_std, 100)
        y_vals = norm.pdf(x_vals, pitch_mean, pitch_std)
    else:
        x_vals = np.array([pitch_mean])  
        y_vals = np.array([1.0])        

    sns.lineplot(x=x_vals, y=y_vals, ax=ax, color="blue", label="Pitch Distribution")
    ax.axvspan(100, 600, color='green', alpha=0.2, label="Ideal Range (100-300 Hz)")
    ax.axvline(pitch_mean, color='red', linestyle='--', linewidth=2, label=f'Mean Pitch: {pitch_mean:.2f} Hz')

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.set_title("Bell Curve of Pitch")
    ax.legend()
    st.pyplot(fig)

    ### **🧹 Clean up Temporary Files**
    os.remove(file_path)

    ### **📌 Footer**
    st.markdown("""
    <style>
        .bottom-right {
            position: fixed;
            bottom: 20px;
            right: 20px;
            font-size: 12px;
            color: #555;
            background-color: rgba(255, 255, 255, 0.7);
            padding: 5px;
            border-radius: 5px;
        }
    </style>
    <div class="bottom-right">
        Developed by Yash Sharma
    </div>
    """, unsafe_allow_html=True)
