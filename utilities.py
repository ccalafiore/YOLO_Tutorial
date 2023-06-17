
import os
import pathlib
import cv2
from distinctipy import distinctipy
import numpy as np
from ultralytics import YOLO
import calapy as cp
import torch


def load_yolo(dir_model):

    """
    :param dir_model:
    :type dir_model: str
    :return:
    :rtype: YOLO
    """

    if not os.path.exists(dir_model):
        dir_root, name_yolo = os.path.split(dir_model)
        os.makedirs(dir_root, exist_ok=True)

    yolo = YOLO(dir_model)

    return yolo


def load_yolos(dirs_models, n_workers=None):

    """

    :param dirs_models:
    :type dirs_models: list | tuple | np.ndarray
    :param n_workers:
    :type n_workers: None | int
    :return:
    :rtype: list | tuple | np.ndarray
    """

    n_yolos = len(dirs_models)
    kwargs = [dict(dir_model=dirs_models[i]) for i in range(0, n_yolos, 1)]

    yolos = cp.threading.MultiThreads(
        func=load_yolo, args=None, kwargs=kwargs, n_workers=n_workers, names=None).run()

    return yolos


def download_yolo(name, dir_root=None):

    """

    :param name:
    :type name: str
    :param dir_root:
    :type dir_root: None| str
    :return:
    :rtype: YOLO
    """

    if dir_root is None:
        dir_root = os.getcwd()
    else:
        os.makedirs(dir_root, exist_ok=True)

    dir_model = os.path.join(dir_root, name)

    suffix = pathlib.Path(dir_model).suffix
    if len(suffix) == 0:
        dir_model += '.pt'

    if os.path.exists(dir_model):
        os.remove(dir_model)

    yolo = YOLO(dir_model)

    return yolo


def download_yolos(names, dirs_roots=None, n_workers=None):

    """

    :param names:
    :type names: list | tuple | np.ndarray
    :param dirs_roots:
    :type dirs_roots: None| list | tuple | np.ndarray
    :param n_workers:
    :type n_workers: None | int
    :return:
    :rtype: list | tuple | np.ndarray
    """

    n_yolos = len(names)
    if dirs_roots is None:
        kwargs = [dict(name=names[i]) for i in range(0, n_yolos, 1)]
    else:
        kwargs = [dict(name=names[i], dir_root=dirs_roots[i]) for i in range(0, n_yolos, 1)]

    yolos = cp.threading.MultiThreads(
        func=download_yolo, args=None, kwargs=kwargs, n_workers=n_workers, names=None).run()

    return yolos


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
            dir_root_out_image, name_out_image = os.path.split(kwargs['dir_out_image'])
            os.makedirs(name=dir_root_out_image, exist_ok=True)
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
