
import os
import utilities


dir_script = __file__
dir_project = os.path.dirname(dir_script)
dir_data = os.path.join(dir_project, 'data')
dir_yolos = os.path.join(dir_data, 'yolos')

names_yolo_versions = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
n_yolos = len(names_yolo_versions)
dirs_roots = [dir_yolos for v in range(0, n_yolos, 1)]

utilities.download_yolos(names=names_yolo_versions, dirs_roots=dirs_roots, n_workers=None)
