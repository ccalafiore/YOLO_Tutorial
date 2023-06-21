
# todo

import os
import utilities


root_script = __file__
root_project = os.path.dirname(root_script)
root_data = os.path.join(root_project, 'data')

root_yolos = os.path.join(root_data, 'yolos')
name_yolo_versions = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
v = 0
name_yolo_version_v = name_yolo_versions[v]
dir_yolo_version_v = os.path.join(root_yolos, name_yolo_version_v + '.pt')

yolo_v = utilities.load_yolo(dir_model=dir_yolo_version_v)

camera_index = 0

dir_out_video = 'output_video.mp4'

box_drawer = utilities.BoxDrawer(names=yolo_v.model.names, colors=None, threshold=(255 * .3))

size = 1280, 720
fps = 5.0

quitting_key = 'q'
timeout = 20  # in secs

utilities.detect_video(
    model=yolo_v, source=camera_index, dir_out_video=dir_out_video, show=True, size=size, fps=fps,
    box_drawer=box_drawer, do_track=False, do_count=False, timeout=timeout, quit_key=quitting_key)
