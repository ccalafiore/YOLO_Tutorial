
import os
import cv2
from distinctipy import distinctipy
import numpy as np
from ultralytics import YOLO
import calapy as cp
import torch


def load_yolo(dir_file):

    """
    :param dir_file:
    :type dir_file: str
    :return:
    :rtype: YOLO
    """

    if os.path.exists(dir_file):
        yolo = YOLO(dir_file)
    else:
        raise FileNotFoundError()

    return yolo


def load_yolos(dirs_files):

    """
    :param dirs_files:
    :type dirs_files: list | tuple | np.ndarray
    :return:
    :rtype: list | tuple | np.ndarray
    """

    # yolos = None
    # return yolos
    raise NotImplementedError()


def download_yolo(name, dir_file=None):

    """

    :param name:
    :type name: str
    :param dir_file:
    :type dir_file: None| str
    :return:
    :rtype: YOLO
    """

    tmp_dir_root = os.getcwd()
    tmp_dir_model = os.path.join(tmp_dir_root, name)

    if os.path.exists(tmp_dir_model):
        os.remove(tmp_dir_model)

    yolo = YOLO(name)

    if dir_file is not None:
        if os.path.exists(dir_file):
            os.remove(dir_file)
        os.rename(tmp_dir_model, dir_file)

    return yolo


def download_yolos(names, dirs_files=None):

    """

    :param names:
    :type names: list | tuple | np.ndarray
    :param dirs_files:
    :type dirs_files: None| list | tuple | np.ndarray
    :return:
    :rtype: list | tuple | np.ndarray
    """

    # yolos = None
    # return yolos
    raise NotImplementedError()


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
            self.threshold = 255 * 0.3
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

    def draw_multi_boxes_in_single_image(self, input_image, xyxy, **kwargs):

        """

        :param input_image:
        :type input_image: str | np.ndarray | torch.Tensor

        :param xyxy:
        :type xyxy: list | tuple | np.ndarray | torch.Tensor

        :key classes:
        :key confidences:
        :key ids:
        :key dir_out_image:

        :return:
        :rtype: np.ndarray | torch.Tensor
        """

        if isinstance(input_image, str):
            image = cv2.imread(input_image)
        else:
            image = input_image

        n_boxes = xyxy.shape[0]

        xyxy_int = cp.maths.round_to_closest_int(xyxy)
        if isinstance(xyxy_int, (np.ndarray, torch.Tensor)):
            xyxy_int = xyxy_int.tolist()

        if kwargs.get('classes') is None:
            classes_int = None  # type: None | list | tuple | np.ndarray | torch.Tensor
        else:
            classes_int = cp.maths.round_to_closest_int(kwargs['classes'])
            if isinstance(classes_int, (np.ndarray, torch.Tensor)):
                classes_int = classes_int.tolist()

        if kwargs.get('confidences') is None:
            confidences_int = None  # type: None | list | tuple | np.ndarray | torch.Tensor
        else:
            confidences_int = cp.maths.round_to_closest_int(kwargs['confidences'] * 100)
            if isinstance(confidences_int, (np.ndarray, torch.Tensor)):
                confidences_int = confidences_int.tolist()

        if kwargs.get('ids') is None:
            ids_int = None  # type: None | list | tuple | np.ndarray | torch.Tensor
        else:
            ids_int = cp.maths.round_to_closest_int(kwargs['ids'])
            if isinstance(ids_int, (np.ndarray, torch.Tensor)):
                ids_int = ids_int.tolist()

        for b in range(0, n_boxes, 1):

            if classes_int is None:
                class_b = None
                name_b = None
                color_b = self.colors[0]
                text_colors_b = self.text_colors[0]
            else:
                class_b = classes_int[b]
                name_b = self.names[class_b]
                color_b = self.colors[class_b]
                text_colors_b = self.text_colors[class_b]

            if confidences_int is None:
                confidence_b = None
            else:
                confidence_b = confidences_int[b]

            if ids_int is None:
                id_b = None
            else:
                id_b = ids_int[b]

            if (name_b is None) and (confidence_b is None) and (id_b is None):
                label_b = None

            elif (name_b is None) and (confidence_b is None) and (id_b is not None):
                label_b = 'ID: {ID:d}'.format(ID=id_b)

            elif (name_b is None) and (confidence_b is not None) and (id_b is None):
                label_b = '{confidence:d}%'.format(confidence=confidence_b)

            elif (name_b is None) and (confidence_b is not None) and (id_b is not None):
                label_b = '{confidence:d}% - ID: {ID:d}'.format(confidence=confidence_b, ID=id_b)

            elif (name_b is not None) and (confidence_b is None) and (id_b is None):
                label_b = name_b

            elif (name_b is not None) and (confidence_b is None) and (id_b is not None):
                label_b = '{name:s} - ID: {ID:d}'.format(name=name_b, ID=id_b)

            elif (name_b is not None) and (confidence_b is not None) and (id_b is None):
                label_b = '{name:s}: {confidence:d}%'.format(name=name_b, confidence=confidence_b)
            else:
                label_b = '{name:s}: {confidence:d}% - ID: {ID:d}'.format(name=name_b, confidence=confidence_b, ID=id_b)

            x1, y1, x2, y2 = xyxy_int[b]
            cv2.rectangle(image, (x1, y1), (x2, y2), color_b, 2)

            if label_b is not None:
                # Get the size of the text
                text_size_b = cv2.getTextSize(label_b, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                # Draw a filled rectangle to put the text on
                cv2.rectangle(image, (x1, y1 - text_size_b[1] - 4), (x1 + text_size_b[0] + 4, y1), color_b, -1)

                # Put the text on the image
                cv2.putText(image, label_b, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colors_b, 1)

        if kwargs.get('dir_out_image') is not None:
            cv2.imwrite(filename=kwargs['dir_out_image'], img=image)

        return image

    def draw_multi_boxes_in_multi_images(self, input_images, xyxy, n_workers=None, **kwargs):

        """

        :param input_images:
        :type input_images: list | tuple | np.ndarray

        :param xyxy:
        :type xyxy: list | tuple | np.ndarray

        :arg n_workers:
        :type n_workers: int | None

        :key classes:
        :key confidences:
        :key ids:
        :key dir_out_images:

        :return:
        :rtype: list | tuple | np.ndarray
        """

        I = len(input_images)

        if 'dir_out_images' in kwargs.keys():
            kwargs['dir_out_image'] = kwargs.pop('dir_out_images')

        kwargs_next = [{
            'input_image': input_images[i], 'xyxy': xyxy[i], **{key_k: kwargs[key_k][i] for key_k in kwargs.keys()}}
            for i in range(0, I, 1)]

        out_images = cp.threading.MultiThreads(
            func=self.draw_multi_boxes_in_single_image,
            args=None, kwargs=kwargs_next, n_workers=n_workers, names=None).run()

        if isinstance(input_images, list):
            return out_images
        elif isinstance(input_images, tuple):
            return tuple(out_images)
        elif isinstance(input_images, np.ndarray):
            return np.asarray(out_images)

    __call__ = draw_multi_boxes_in_multi_images
