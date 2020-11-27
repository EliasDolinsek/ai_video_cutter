# ai_video_cutter

## Idea
Creating an AI which can cut videos to music would be cool.

## Problem
This project uses pyAudioAnalysis to analyse sound files and PySceneDetect to detect cuts in the associated video file.
The data is then being used to train a TensorFlow model. Testing the model resulted in a loss of NaN and an accuracy 
of 0.0 which indicates that it is not possible to decide if a cut should be made, purely by the data provided by the two libraries.

## Features
* Download YouTube videos specified in videos_list.csv
* Split the downloaded file into audio and video
* Analys audio and video and save data into the data folder
* Train TensorFlow model using data stored in data folder
