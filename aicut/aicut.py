import sys, os
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import pandas as pd
import numpy as np
import tensorflow as tf

WINDOW_VALUE = 0.05
STEP_VALUE = 0.025


def calculate_window(frame_rate):
    return WINDOW_VALUE * frame_rate


def calculate_step(frame_rate):
    return STEP_VALUE * frame_rate


def calculate_begin_timestamps(frames_count):
    return np.array([[i*STEP_VALUE for i in range(frames_count)]])


def calculate_end_timestamps(frames_count):
    timestamps = calculate_begin_timestamps(frames_count)
    for i in range(len(timestamps[0])):
        timestamps[0, i] += WINDOW_VALUE

    return timestamps


def analyze_audio(file):
    [fs, x] = audioBasicIO.read_audio_file(file)
    x = audioBasicIO.stereo_to_mono(x)

    step_duration = calculate_step(fs)
    f, f_names = ShortTermFeatures.feature_extraction(x, fs, calculate_window(fs), step_duration)
  
    result = f.reshape(-1, f.shape[0])

    begin_timestamps = calculate_begin_timestamps(f.shape[1])
    end_timestamps = calculate_end_timestamps(f.shape[1])

    df = pd.DataFrame(data=result, columns=f_names)
    df["start_timestamp"] = begin_timestamps[0]
    df["end_timestamp"] = end_timestamps[0]
   
    return df


def find_cuts(audio_analysis):
    cuts_start_timestamp = []
    cuts_end_timestamp = []


    model = tf.keras.models.load_model("model.h5")
    for row in audio_analysis.iterrows():
        del audio_analysis["start_timestamp"]
        del audio_analysis["end_timestamp"]
        
        model.predict(row)
        
    return audio_analysis[audio_analysis.cut == 1]

    
if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        print("Missing arguments. Format music_file videos_folder output.csv")
        exit()

    audio_file = args[1]
    videos_filder = args[2]
    output_file = args[3]

    audio_result = analyze_audio(audio_file)
    print(audio_result.head())
    # print(filter_cuts(audio_result))

    
