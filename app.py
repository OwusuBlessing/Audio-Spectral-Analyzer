import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from visuals import plot_frequency_domain, plot_spectrogram, plot_waveform,compute_fourier
from scipy.signal import butter, filtfilt

# Function to apply low-pass filter
def apply_lowpass_filter(audio_data, sr, cutoff_freq, order=5):
    nyquist_freq = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_audio = filtfilt(b, a, audio_data)
    return filtered_audio

# Main code


# Main code
st.sidebar.title("About")
st.sidebar.write("""
The Audio Spectral Analyzer is a powerful tool for visualizing and analyzing audio signals. 
Whether you're a musician, sound engineer, researcher, or simply an audio enthusiast, 
this app offers a range of features to explore and understand the spectral characteristics of audio files.

**Key Features:**

- **Upload and Playback**: Upload your own audio files in popular formats such as MP3, WAV, or OGG, and listen to them with playback controls.
- **Visualizations**: Explore the frequency content of your audio with interactive visualizations, including waveform, frequency domain plot, and spectrogram.
- **Effects and Processing**: Apply a variety of audio effects and processing techniques to modify and enhance your audio, such as pitch shifting, time stretching, reverb, echo, and fade in/out.
- **Spectral Analysis**: Utilize Fourier transforms to analyze the frequency content of your audio signals, gaining insights into their spectral characteristics.

**Why Use This App?**

- **Educational**: Learn about audio processing techniques and spectral analysis in an interactive and intuitive environment.
- **Creative**: Experiment with different effects and processing to create unique sounds and audio effects.
- **Professional**: Analyze and enhance audio files for music production, sound design, research, and more.

Start exploring the fascinating world of audio spectral analysis with the Audio Spectral Analyzer app!
""")


st.title("Audio Spectral Analyzer")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

if uploaded_file is not None:
    # Read the uploaded audio file
    audio_data, sr = librosa.load(uploaded_file, sr=None)

    # Display audio properties
    st.write("### Audio properties")
    audio_duration = librosa.get_duration(y=audio_data, sr=sr)
    st.write(f"Audio Duration: {audio_duration:.2f} seconds")
    st.write(f"Sample Rate: {sr} Hz")
    st.write(f"Number of Channels: {1 if len(audio_data.shape) == 1 else audio_data.shape[0]}")

    # Display audio player with playback controls
    audio_player = st.audio(audio_data, format='audio/wav', start_time=0, sample_rate=sr)
   
    # Visualization section
    st.write("### Visualizations")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Plot Waveform"):
            plot_waveform(audio_data, sr)

    with col2:
        if st.button("Plot Frequency Domain"):
            plot_frequency_domain(audio_data, sr)

    with col3:
        if st.button("Plot Spectrogram"):
            plot_spectrogram(audio_data, sr)

    # Effects section
    st.write("### Effects")
    cutoff_freq = st.slider("Cutoff Frequency (Hz)", min_value=20, max_value=10000, value=1000, step=100)
    if st.button("Apply Low-Pass Filter"):
        filtered_audio = apply_lowpass_filter(audio_data, sr, cutoff_freq)
        st.audio(filtered_audio, format='audio/wav', sample_rate=sr)
        st.success("Low-pass filter applied successfully!")
    
    
    # Pitch shifting
    pitch_shift = st.slider("Pitch Shift (semitones)", min_value=-12, max_value=12, value=0, step=1)
    if st.button("Apply Pitch Shifting"):
        shifted_audio = librosa.effects.pitch_shift(audio_data, sr, n_steps=pitch_shift)
        st.audio(shifted_audio, format='audio/wav', sample_rate=sr)
        st.success(f"Pitch shifted by {pitch_shift} semitones!")

    # Time stretching
    time_stretch = st.slider("Time Stretch (%)", min_value=50, max_value=200, value=100, step=10)
    if st.button("Apply Time Stretching"):
        stretched_audio = librosa.effects.time_stretch(audio_data, time_stretch / 100)
        st.audio(stretched_audio, format='audio/wav', sample_rate=sr)
        st.success(f"Time stretched by {time_stretch}%!")

    # Reverb
    reverb_amount = st.slider("Reverb Amount", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    if st.button("Apply Reverb"):
        reverberated_audio = librosa.effects.preemphasis(audio_data, coef=reverb_amount)
        st.audio(reverberated_audio, format='audio/wav', sample_rate=sr)
        st.success("Reverb applied successfully!")

    # Echo
    echo_delay = st.slider("Echo Delay (s)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    echo_decay = st.slider("Echo Decay", min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    if st.button("Apply Echo"):
        echo_audio = librosa.effects.preemphasis(audio_data, coef=echo_delay)
        st.audio(echo_audio, format='audio/wav', sample_rate=sr)
        st.success("Echo applied successfully!")