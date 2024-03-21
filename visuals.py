
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

# Function to compute Fourier transform
def compute_fourier(audio_data, sr):
    n_fft = 2048  # Number of FFT points
    fft_result = np.fft.fft(audio_data, n_fft)
    frequencies = np.fft.fftfreq(n_fft, d=1/sr)
    return frequencies, np.abs(fft_result)

# Function to plot time domain (waveform)
def plot_waveform(audio_data, sr):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(audio_data)) / sr, audio_data)
    ax.set_title('Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid()
    st.pyplot(fig)

# Function to plot frequency domain
def plot_frequency_domain(audio_data, sr):
    frequencies, fft_result = compute_fourier(audio_data, sr)
    fig, ax = plt.subplots()
    ax.plot(frequencies, fft_result)
    ax.set_title('Frequency Domain (Fourier Transform)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid()
    st.pyplot(fig)

# Function to plot spectrogram
def plot_spectrogram(audio_data, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    fig, ax = plt.subplots()
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(fig)