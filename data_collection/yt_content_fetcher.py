import os
import subprocess
import pandas as pd
from pytube import YouTube
from audio_analyzer import analyze_audio

DATA_STORAGE_PATH = "data"
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
for index, row in videos_list.iterrows():
    current_file = row.values[0]
    output_file_name = f"video_{index}"

    try:
        file = download_video(current_file, output_file_name)
        audio_file = extract_audio(file, output_file_name)
    except:
        print("An failure occurred during content fetching - skipping video!")
        continue

    print(file)
    print("Analyzing audio...")
    analyze_audio(audio_file)
    print("Finished analyzing audio")

print("Finished task!")
