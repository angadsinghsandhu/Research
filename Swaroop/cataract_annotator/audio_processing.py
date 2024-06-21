import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd

# Global variables
audio_data = []
samplerate = 44100
channels = 2

def audio_callback(indata, frames, time, status):
    global audio_data
    audio_data.append(indata.copy())

def save_audio(output_audio_path):
    global audio_data, samplerate
    audio_array = np.concatenate(audio_data, axis=0)
    write(output_audio_path, samplerate, audio_array)
