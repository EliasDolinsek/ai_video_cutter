# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

def find_cuts(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor()

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scenes = scene_manager.get_scene_list(base_timecode)

    return map(lambda cut: cut[1].get_seconds(), scenes)

