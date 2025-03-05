import os
os.system('apt-get update')
os.system('apt-get install -y ffmpeg')
os.system('apt-get install -y ffprobe')

import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft
from scipy.stats import norm
from pydub import AudioSegment
from transformers import pipeline
import os

# Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit UI
st.title("\U0001F3A4 Single Audio Sentiment Analysis")
st.write("Upload an MP3 file to analyze its sentiment.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

def analyze_audio(file_path):
    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(file_path)
    wav_path = file_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    
    # Load audio
    y, sr = librosa.load(wav_path, sr=None)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Compute FFT
    fft_vals = np.abs(fft(y))[:len(y)//2]
    freqs = np.linspace(0, sr/2, len(fft_vals))
    peak_freq = freqs[np.argmax(fft_vals)]
    
    # Estimate pitch distribution (Bell curve)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean, pitch_std = np.mean(pitch_values), np.std(pitch_values)
    
    # Determine if peak frequency is high or low
    threshold = 300  # Example threshold for high pitch detection
    peak_color = 'red' if peak_freq > threshold else 'green'
    
    # Dummy sentiment analysis
    sentiment_result = sentiment_analyzer("This is a placeholder for sentiment analysis based on audio!")
    
    os.remove(wav_path)  # Clean up
    
    return sentiment_result[0], mfccs_mean, freqs, fft_vals, pitch_mean, pitch_std, peak_freq, peak_color

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    sentiment, mfccs, freqs, fft_vals, pitch_mean, pitch_std, peak_freq, peak_color = analyze_audio(file_path)
    
    st.subheader("\U0001F4CA Sentiment Analysis Result")
    st.write(f"**Sentiment:** {sentiment['label']}")
    st.write(f"**Confidence:** {sentiment['score']:.2f}")
    
    # Explanation blocks
    st.markdown("### What is MFCC and Why is it Important?")
    st.write("MFCC (Mel-Frequency Cepstral Coefficients) helps analyze the quality and tone of speech. It is widely used in speech recognition and emotion detection.")
    
    # Color coding for MFCC evaluation
    mfcc_quality = 'Good' if np.mean(mfccs) > -100 else 'Bad'
    mfcc_color = 'green' if mfcc_quality == 'Good' else 'red'
    st.markdown(f"**MFCC Quality:** <span style='color:{mfcc_color}'>{mfcc_quality}</span>", unsafe_allow_html=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MFCC plot
    axes[0].bar(range(len(mfccs)), mfccs)
    axes[0].set_xlabel("MFCC Coefficients")
    axes[0].set_ylabel("Mean Value")
    axes[0].set_title("MFCC Feature Extraction")
    
    # Explanation for FFT
    st.markdown("### What is FFT and Why is it Important?")
    st.write("FFT (Fast Fourier Transform) helps analyze the frequency content of the voice, which can indicate pitch and clarity.")
    
    # Color coding for FFT peak frequency evaluation
    fft_quality = 'High' if peak_freq > 300 else 'Normal'
    fft_color = 'red' if fft_quality == 'High' else 'green'
    st.markdown(f"**FFT Peak Frequency:** <span style='color:{fft_color}'>{peak_freq:.2f} Hz ({fft_quality})</span>", unsafe_allow_html=True)
    
    # FFT plot
    axes[1].plot(freqs, fft_vals, label='FFT Spectrum')
    axes[1].axvline(peak_freq, color=peak_color, linestyle='--', label=f'Peak: {peak_freq:.2f} Hz')
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("FFT of Voice Data")
    axes[1].legend()
    
    st.pyplot(fig)
    
    # Explanation for pitch
    st.markdown("### What is Pitch Distribution and Why is it Important?")
    st.write("Pitch distribution helps identify the general pitch range of the voice, which is crucial in customer service calls to analyze tone and engagement.")
    
    # Color coding for pitch evaluation
    pitch_quality = 'Good' if 100 < pitch_mean < 300 else 'Bad'
    pitch_color = 'green' if pitch_quality == 'Good' else 'red'
    st.markdown(f"**Pitch Quality:** <span style='color:{pitch_color}'>{pitch_mean:.2f} Hz ({pitch_quality})</span>", unsafe_allow_html=True)
    
    # Pitch Bell Curve Plot
    st.subheader("🎵 Pitch Distribution")
    fig, ax = plt.subplots()
    x_vals = np.linspace(pitch_mean - 3*pitch_std, pitch_mean + 3*pitch_std, 100)
    y_vals = norm.pdf(x_vals, pitch_mean, pitch_std)
    sns.lineplot(x=x_vals, y=y_vals, ax=ax)
    ax.axvline(pitch_mean, color='blue', linestyle='--', label=f'Mean Pitch: {pitch_mean:.2f} Hz')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.set_title("Bell Curve of Pitch")
    ax.legend()
    st.pyplot(fig)
    
    os.remove(file_path)
