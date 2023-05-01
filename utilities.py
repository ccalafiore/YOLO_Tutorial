
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


