
import os
import utilities


root_script = __file__
root_project = os.path.dirname(root_script)
root_data = os.path.join(root_project, 'data')

root_yolos = os.path.join(root_data, 'yolos')

names_yolo_versions = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
n_yolos = len(names_yolo_versions)
dirs_yolos = [os.path.join(root_yolos, names_yolo_versions[v]) for v in range(0, n_yolos, 1)]

utilities.download_yolos(dirs_models=dirs_yolos, n_workers=None)
