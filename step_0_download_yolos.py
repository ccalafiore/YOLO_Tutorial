
import os
import utilities
import calapy as cp


dir_script = __file__
dir_project = os.path.dirname(dir_script)

dir_data = os.path.join(dir_project, 'data')

dir_yolos = os.path.join(dir_data, 'yolos')
os.makedirs(dir_yolos, exist_ok=True)

name_yolo_versions = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
dir_yolo_versions = [os.path.join(dir_yolos, name_v) for name_v in name_yolo_versions]

n_yolos = len(name_yolo_versions)

kwargs = [dict(name=name_yolo_versions[i], dir_file=dir_yolo_versions[i]) for i in range(0, n_yolos, 1)]

yolos = cp.threading.MultiThreads(
    func=utilities.download_yolo, args=None, kwargs=kwargs, n_workers=None, names=None).run()
