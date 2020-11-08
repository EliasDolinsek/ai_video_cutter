import os
import pathlib
import subprocess

import pandas as pd
from pytube import YouTube

from audio_analyzer import analyze_audio
from video_analyzer import find_cuts

DATA_STORAGE_PATH = "temp"
videos_list = pd.read_csv("videos_list.csv", header=None)


def download_video(url, output_file_name):
    print("Fetching data for YouTube-Video with url", url, "...")

    video = YouTube(url)
    video_stream = video.streams.get_highest_resolution()
    print("Fetched data for", video.title, "successfully")

    print("Downloading video...")
    download_result = video_stream.download(DATA_STORAGE_PATH, filename=output_file_name)
    print("Downloaded and saved video in", DATA_STORAGE_PATH)

    return download_result


def extract_audio(file, output_file_name):
    print("Extracting audio")

    output_file_path = os.path.join(os.path.dirname(file), f"{output_file_name}.wav")
    command = f"ffmpeg -i {file} -vn -acodec pcm_s16le -ar 44100 -ac 2 {output_file_path}"

    subprocess.call(command, shell=True)
    print("Audio extracted successfully")

    return output_file_path


print("Starting data collection...")
current_path = pathlib.Path(__file__).parent.absolute()
for index, row in videos_list.iterrows():
    current_file = row.values[0]
    output_file_name = f"video_{index}"

    try:
        video_file = download_video(current_file, output_file_name)
        audio_file = extract_audio(video_file, output_file_name)
    except:
        print("An failure occurred during content fetching - skipping video!")
        continue

    print("Analyzing audio...")
    output_path = os.path.join(current_path, f"data/{output_file_name}.csv")
    data = analyze_audio(audio_file)
    print("Finished analyzing audio")

    print("Analyzing video...")
    data["cut"] = [0 for i in range(len(data))]
    scenes = find_cuts(video_file)

    for scene_cut_timestamp in scenes:
        for row_index in data.index:
            if data["start_timestamp"][row_index] < scene_cut_timestamp < data["end_timestamp"][row_index]:
                data.at[row_index, "cut"] = 1

    base_path = os.path.abspath(output_path + "/../")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    data.to_csv(output_path)
    print("Finished analyzing video")

    # delete temp files
    os.remove(video_file)
    os.remove(audio_file)

print("Finished task!")
