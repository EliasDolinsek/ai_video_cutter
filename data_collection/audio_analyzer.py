from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import pandas as pd
import numpy as np

WINDOW_VALUE = 0.05
STEP_VALUE = 0.025


def calculate_window(frame_rate):
    return WINDOW_VALUE * frame_rate


def calculate_step(frame_rate):
    return STEP_VALUE * frame_rate


def calculate_timestamps(frames_count):
    for i in range(frames_count):
        print(i*STEP_VALUE)
    return np.array([[i*STEP_VALUE for i in range(frames_count)]])


def analyze_audio(file):
    [fs, x] = audioBasicIO.read_audio_file(file)
    x = audioBasicIO.stereo_to_mono(x)

    step_duration = calculate_step(fs)
    f, f_names = ShortTermFeatures.feature_extraction(x, fs, calculate_window(fs), step_duration)
  
    result = f.reshape(-1, f.shape[0])

    begin_timestamps = calculate_timestamps(f.shape[1])
    df = pd.DataFrame(data=result, columns=f_names)
    df["start_timestamp"] = begin_timestamps[0]
    
    df.to_csv("test.csv")


analyze_audio("test.wav")