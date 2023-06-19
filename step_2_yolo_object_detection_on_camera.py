
# todo

import os
import math
import time
import keyboard
import numpy as np
import torch
import utilities
import calapy as cp
import cv2


def detect_video(
        model, source, dir_out_video=None, show=False, size=None, fps=None,
        do_track=False, do_count=False, timeout=None, quit_key=None):

    if isinstance(model, str):
        model = utilities.load_yolo(dir_model=model)

    if isinstance(source, int):
        cap = cv2.VideoCapture(index=source)
    elif isinstance(source, str):
        cap = cv2.VideoCapture(filename=source)
    else:
        raise TypeError('source')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    if size is None:
        size = w, h = cap.get(3), cap.get(4)
    else:
        if isinstance(size, int):
            size = tuple([size, size])
        elif isinstance(size, list):
            size = tuple(size)
        elif isinstance(size, tuple):
            pass
        elif isinstance(size, (np.ndarray, torch.Tensor)):
            size = tuple(size.tolist())
        else:
            raise TypeError('size')

        w, h = size
        cap.set(3, w)
        cap.set(4, h)

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    # if fps is None:
    #     fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
    # elif isinstance(fps, int):
    #     pass
    # elif isinstance(fps, float):
    #     fps = math.floor(fps)
    # else:
    #     raise TypeError('fps')

    mspf = math.floor(1000 / fps)
    spf = mspf / 1000

    if timeout is None:
        timeout = math.inf

    if quit_key is None:
        pass
    elif isinstance(quit_key, (str, int)):
        pass
    else:
        raise TypeError('quit_key')

    if isinstance(source, int):
        # start the model by processing one image
        success, image = cap.read()
        if success:
            results = model(source=image, stream=False, verbose=False)

    playing = True

    timer = cp.clock.Timer()

    while playing:

        success, image = cap.read()

        if success:

            results = model(source=image, stream=False, verbose=False)

            xyxy = results[0].boxes.xyxy
            classes = results[0].boxes.cls
            confidences = results[0].boxes.conf

            image = box_drawer.draw_multi_boxes_in_single_image(
                input_image=image, xyxy=xyxy, classes=classes, confidences=confidences, dir_out_image=None)

            if show:
                cv2.imshow(winname='Image', mat=image)
                cv2.pollKey()
                if keyboard.is_pressed(quit_key) or (timer.get_seconds() >= timeout):
                    playing = False
                else:
                    # todo replace cv2.pollKey() with time.sleep() or cp.clock.wait()
                    # cv2.waitKey(delay=mspf)
                    time.sleep(spf)
            else:
                if keyboard.is_pressed(quit_key) or (timer.get_seconds() >= timeout):
                    playing = False
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            playing = False

    cap.release()
    cv2.destroyAllWindows()

    return None


root_script = __file__
root_project = os.path.dirname(root_script)
root_data = os.path.join(root_project, 'data')

root_yolos = os.path.join(root_data, 'yolos')
name_yolo_versions = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
v = 0
name_yolo_version_v = name_yolo_versions[v]
dir_yolo_version_v = os.path.join(root_yolos, name_yolo_version_v + '.pt')

yolo_v = utilities.load_yolo(dir_model=dir_yolo_version_v)

box_drawer = utilities.BoxDrawer(names=yolo_v.model.names, colors=None, threshold=(255 * .3))

camera_index = 1

size = 1289, 720
fps = 30

quitting_key = 'q'
timeout = None  # in secs

detect_video(
    model=yolo_v, source=camera_index, dir_out_video=None, show=True, size=size, fps=fps,
    do_track=False, do_count=False, timeout=timeout, quit_key=quitting_key)
