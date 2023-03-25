
import os
from ultralytics import YOLO
import cv2
from distinctipy import distinctipy
import numpy as np


class BoxDrawer:

    def __init__(self, names, colors=None, threshold=None):

        self.names = names
        self.K = len(self.names)

        if colors is None:
            # generate N visually distinct colours
            self.colors = [
                tuple(color_c[::-1]) for color_c in
                (np.asarray(distinctipy.get_colors(self.K), dtype='f') * 225).astype('i').tolist()]
        else:
            self.colors = colors

        self.white = (225, 225, 255)
        self.black = (0, 0, 0)

        if threshold is None:
            self.threshold = 225 * .5
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

    def __call__(self, yolo_results, image):

        boxes = yolo_results[0].boxes

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

        return image


if __name__ == '__main__':

    dir_script = __file__
    dir_step = os.path.dirname(dir_script)
    dir_project = os.path.dirname(dir_step)
    dir_data = os.path.join(dir_project, 'data')

    name_images = [
        'bus_rapid_transit.jpg', 'people_cars_bicycles_1.jpg', 'people_cars_bicycles_2.jpg', 'dino_1.jpg',
        'amici_dispersi.jpg', 'trial.jpg', 'dinner.jpg']

    i_image = 6
    name_image_i = name_images[i_image]
    dir_images = os.path.join(dir_data, 'images')
    dir_input_images = os.path.join(dir_images, 'input_images')
    dir_image_i = os.path.join(dir_input_images, name_image_i)

    dir_yolo_parameters = os.path.join(dir_data, 'yolo_parameters')
    name_yolo_versions = ['yolov8n.pt', 'yolov8l.pt']
    v = 1
    name_yolo_version_v = name_yolo_versions[v]
    dir_yolo_version_v = os.path.join(dir_yolo_parameters, name_yolo_version_v)

    # todo: download yolo if it does not exist
    # if not os.path.exists(dir_yolo_version_v):
    #     cp.ml.models.yolo.download()

    yolo_v = YOLO(dir_yolo_version_v)

    results = yolo_v(source=dir_image_i)

    box_drawer = BoxDrawer(names=yolo_v.model.names, colors=None, threshold=(225 * .5))

    image_i = cv2.imread(dir_image_i)

    box_drawer(yolo_results=results, image=image_i)

    dir_out_images = os.path.join(dir_images, 'output_images')
    os.makedirs(name=dir_out_images, exist_ok=True)

    dir_out_image_i = os.path.join(dir_out_images, name_image_i)
    cv2.imwrite(filename=dir_out_image_i, img=image_i)
