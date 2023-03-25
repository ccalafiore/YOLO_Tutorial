
import os
from ultralytics import YOLO


def download(name_model, dir_file=None):

    tmp_dir_root = os.getcwd()
    tmp_dir_model = os.path.join(tmp_dir_root, name_model)

    if os.path.exists(tmp_dir_model):
        os.remove(tmp_dir_model)

    yolo = YOLO(name_model)

    if dir_file is not None:
        if os.path.exists(dir_file):
            os.remove(dir_file)
        os.rename(tmp_dir_model, dir_file)

    return yolo


if __name__ == '__main__':

    dir_script = __file__
    dir_step = os.path.dirname(dir_script)
    dir_project = os.path.dirname(dir_step)
    dir_data = os.path.join(dir_project, 'data')

    dir_yolo_parameters = os.path.join(dir_data, 'yolo_parameters')
    os.makedirs(dir_yolo_parameters, exist_ok=True)

    name_yolo_versions = ['yolov8n.pt', 'yolov8l.pt']
    v = 1
    name_yolo_version_v = name_yolo_versions[v]

    dir_yolo_version_v = os.path.join(dir_yolo_parameters, name_yolo_version_v)

    download(name_model=name_yolo_version_v, dir_file=dir_yolo_version_v)


