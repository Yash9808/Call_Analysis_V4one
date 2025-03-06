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

# Cache the Sentiment Model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_model()

# Cache the Chartwell Logo Download
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

# Cache the Audio Analysis Function
@st.cache_data
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    fft_vals = np.abs(fft(y))[:len(y)//2]
    freqs = np.linspace(0, sr/2, len(fft_vals))
    peak_freq = freqs[np.argmax(fft_vals)]
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    
    if len(pitch_values) > 0:
        pitch_mean, pitch_std = np.mean(pitch_values), np.std(pitch_values)
    else:
        pitch_mean, pitch_std = 0, 0

    threshold = 300  
    peak_color = 'red' if peak_freq > threshold else 'green'
    sentiment_result = sentiment_analyzer("This is a placeholder for sentiment analysis based on audio!")

    return sentiment_result[0], mfccs_mean, freqs, fft_vals, pitch_mean, pitch_std, peak_freq, peak_color

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    sentiment, mfccs, freqs, fft_vals, pitch_mean, pitch_std, peak_freq, peak_color = analyze_audio(file_path)

    st.subheader("📊 Sentiment Analysis Result")
    st.write(f"**Sentiment:** {sentiment['label']}")
    st.write(f"**Confidence:** {sentiment['score']:.2f}")

    # Pitch Distribution Plot
    st.subheader("🎵 Pitch Distribution")
    fig, ax = plt.subplots()
    
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

    # Clean up
    os.remove(file_path)
