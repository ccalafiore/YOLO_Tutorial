
import os
import cv2
from distinctipy import distinctipy
import numpy as np
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


class BoxDrawer:

    def __init__(self, names, colors=None, threshold=None):

        self.names = names
        self.K = len(self.names)

        if colors is None:
            # generate N visually distinct colours
            self.colors = [
                tuple(color_c[::-1]) for color_c in
                (np.asarray(distinctipy.get_colors(self.K), dtype='f') * 255).astype('i').tolist()]
        else:
            self.colors = colors

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        if threshold is None:
            self.threshold = 255 * 0.5
        else:
            self.threshold = threshold

        self.text_colors = [None for k in range(0, self.K, 1)]  # type: list

        for k, color_k in enumerate(iterable=self.colors, start=0):

            mean_k = sum(color_k) / len(color_k)
            if mean_k < self.threshold:
                color_text_k = self.white
            else:
                color_text_k = self.black

            self.text_colors[k] = color_text_k

    def __call__(self, boxes, input_image, dir_out_image=None):

        if isinstance(input_image, str):
            image = cv2.imread(input_image)
        else:
            image = input_image

        for b, box_b in enumerate(iterable=boxes, start=0):

            class_b = box_b.cls[0].int().tolist()

            x1, y1, x2, y2 = box_b.xyxy[0].int().tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), self.colors[class_b], 2)

            name_b = self.names[class_b]
            conf_b = box_b.conf[0].tolist()
            label_b = f"{name_b}: {conf_b * 100:.0f}%"

            # Get the size of the text
            text_size_b = cv2.getTextSize(label_b, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            # Draw a filled rectangle to put the text on
            cv2.rectangle(image, (x1, y1 - text_size_b[1] - 4), (x1 + text_size_b[0] + 4, y1), self.colors[class_b], -1)

            # Put the text on the image
            cv2.putText(image, label_b, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_colors[class_b], 1)

        if dir_out_image is not None:
            cv2.imwrite(filename=dir_out_image, img=image)

        return image