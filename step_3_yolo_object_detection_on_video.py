

import os
import utilities


root_script = __file__
root_project = os.path.dirname(root_script)
root_data = os.path.join(root_project, 'data')

root_yolos = os.path.join(root_data, 'yolos')
name_yolo_versions = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
v = 4
name_yolo_version_v = name_yolo_versions[v]
dir_yolo_version_v = os.path.join(root_yolos, name_yolo_version_v + '.pt')

yolo_v = utilities.load_yolo(dir_model=dir_yolo_version_v)

name_input_video = 'highway_0.mp4'
root_videos = os.path.join(root_data, 'videos')
root_input_videos = os.path.join(root_videos, 'input_videos')
dir_input_video = os.path.join(root_input_videos, name_input_video)

root_out_videos = os.path.join(root_videos, 'output_videos', name_yolo_version_v)
dir_out_video = os.path.join(root_out_videos, 'output_' + name_input_video)

box_drawer = utilities.BoxDrawer(names=yolo_v.model.names, colors=None, threshold=(255 * .3))

size = None
fps = None

quitting_key = 'q'
timeout = None  # in secs

utilities.detect_video(
    model=yolo_v, source=dir_input_video, dir_out_video=dir_out_video, show=False, size=size, fps=fps,
    box_drawer=box_drawer, do_track=False, do_count=False, timeout=timeout, quit_key=quitting_key)
