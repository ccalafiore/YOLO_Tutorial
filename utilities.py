
import os
import math
import keyboard
import pathlib
import cv2
from distinctipy import distinctipy
import numpy as np
import torch
from ultralytics import YOLO
import calapy as cp


def load_yolo(dir_model):

    """
    :param dir_model:
    :type dir_model: str
    :return:
    :rtype: YOLO
    """

    dir_root, name_yolo = os.path.split(dir_model)
    if len(dir_root) == 0:
        dir_root = os.getcwd()
        dir_model = os.path.join(dir_root, name_yolo)
    else:
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


def download_yolo(dir_model):

    """

    :param dir_model:
    :type dir_model: str
    :return:
    :rtype: YOLO
    """

    dir_root, name_yolo = os.path.split(dir_model)
    if len(dir_root) == 0:
        dir_root = os.getcwd()
        dir_model = os.path.join(dir_root, name_yolo)
    else:
        os.makedirs(dir_root, exist_ok=True)

    if os.path.exists(dir_model):
        os.remove(dir_model)

    yolo = YOLO(dir_model)

    return yolo


def download_yolos(dirs_models, n_workers=None):

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
            if len(dir_root_out_image) == 0:
                dir_root_out_image = os.getcwd()
                kwargs['dir_out_image'] = os.path.join(dir_root_out_image, name_out_image)
            else:
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


def detect_video(
        model, source, dir_out_video=None, show=False, size=None, fps=None,
        box_drawer=None, do_track=False, do_count=False, timeout=None, quit_key=None):

    if isinstance(model, str):
        model = load_yolo(dir_model=model)

    if dir_out_video is None:
        format_out_video = None
        write_out_video = False
    else:
        path_out_video = pathlib.Path(dir_out_video)
        if len(path_out_video.suffix) == 0:
            if isinstance(source, int):
                format_out_video = 'mp4'
            else:
                path_source = pathlib.Path(source)
                format_out_video = path_source.suffix.removeprefix('.')
            dir_out_video = '.'.join([dir_out_video, format_out_video])
        else:
            format_out_video = path_out_video.suffix.removeprefix('.')

        dir_root_out_video, name_out_video = os.path.split(dir_out_video)
        if len(dir_root_out_video) == 0:
            dir_root_out_video = os.getcwd()
            dir_out_video = os.path.join(dir_root_out_video, name_out_video)
        else:
            os.makedirs(name=dir_root_out_video, exist_ok=True)

        write_out_video = True

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
        pass
    else:
        if isinstance(size, int):
            size = tuple([size, size])

        if isinstance(size, float):
            tmp = math.floor(size)
            size = tuple([tmp, tmp])

        elif isinstance(size, list):
            size = tuple(cp.maths.round_down_to_closest_int(num=size))

        elif isinstance(size, tuple):
            size = cp.maths.round_down_to_closest_int(num=size)

        elif isinstance(size, (np.ndarray, torch.Tensor)):
            size = tuple(cp.maths.round_down_to_closest_int(num=size).tolist())
        else:
            raise TypeError('size')

        w, h = size
        if w != cap.get(3):
            cap.set(3, w)
        if h != cap.get(4):
            cap.set(4, h)

    size = w, h = cp.maths.round_down_to_closest_int(num=(cap.get(3), cap.get(4)))

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    elif isinstance(fps, (int, float)):
        pass
    else:
        raise TypeError('fps')

    spf = 1 / fps
    mspf = math.floor(spf * 1000)

    if box_drawer is None:
        box_drawer = BoxDrawer(names=model.model.names, colors=None, threshold=(255 * .3))

    if timeout is None:
        timeout = math.inf

    if quit_key is None:
        pass
    elif isinstance(quit_key, (str, int)):
        pass
    else:
        raise TypeError('quit_key')

    if write_out_video:
        if format_out_video == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise ValueError('format_out_video')

        video_writer = cv2.VideoWriter(dir_out_video, fourcc, fps, (w, h), 1)
    else:
        video_writer = None

    timer = None
    t = 0

    capturing = True

    if isinstance(source, int):
        # start the model by processing one image
        success, image = cap.read()
        if success:
            results = model(source=image, stream=False, verbose=False)

    while capturing:

        if keyboard.is_pressed(quit_key):
            print('The Quitting Key was pressed. Quitting ...')
            capturing = False

        elif (t * spf) > timeout:
            print('Timeout')
            capturing = False

        # elif timer.get_seconds_total() >= timeout:
        #     print('Timeout')
        #     capturing = False
        else:

            success, image = cap.read()
            if success:

                results = model(source=image, stream=False, verbose=False)

                xyxy = results[0].boxes.xyxy
                classes = results[0].boxes.cls
                confidences = results[0].boxes.conf

                image = box_drawer.draw_multi_boxes_in_single_image(
                    input_image=image, xyxy=xyxy, classes=classes, confidences=confidences, dir_out_image=None)

                if write_out_video:
                    video_writer.write(image)

                if isinstance(source, int) or show:
                    if timer is None:
                        timer = cp.clock.Timer(ticks_per_sec=fps)
                    else:
                        timer.wait()

                if show:
                    cv2.imshow(winname='Image', mat=image)
                    cv2.pollKey()

                t += 1

            else:
                print('Can\'t receive frame (stream end?). Exiting ...')
                capturing = False

    cv2.destroyAllWindows()
    cap.release()
    if write_out_video:
        video_writer.release()

    return None


def capture_video(source, dir_out_video=None, show=False, size=None, fps=None, timeout=None, quit_key=None):
    """

    :param source:
    :param dir_out_video:
    :param show:
    :param size:
    :param fps:
    :param timeout:
    :param quit_key:
    :return:
    :todo image_transformation:
    """

    if dir_out_video is None:
        format_out_video = None
        write_out_video = False
    else:
        path_out_video = pathlib.Path(dir_out_video)
        if len(path_out_video.suffix) == 0:
            if isinstance(source, int):
                format_out_video = 'mp4'
            else:
                path_source = pathlib.Path(source)
                format_out_video = path_source.suffix.removeprefix('.')
            dir_out_video = '.'.join([dir_out_video, format_out_video])
        else:
            format_out_video = path_out_video.suffix.removeprefix('.')

        dir_root_out_video, name_out_video = os.path.split(dir_out_video)
        if len(dir_root_out_video) == 0:
            dir_root_out_video = os.getcwd()
            dir_out_video = os.path.join(dir_root_out_video, name_out_video)
        else:
            os.makedirs(name=dir_root_out_video, exist_ok=True)

        write_out_video = True

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
        pass
    else:
        if isinstance(size, int):
            size = tuple([size, size])

        if isinstance(size, float):
            tmp = math.floor(size)
            size = tuple([tmp, tmp])

        elif isinstance(size, list):
            size = tuple(cp.maths.round_down_to_closest_int(num=size))

        elif isinstance(size, tuple):
            size = cp.maths.round_down_to_closest_int(num=size)

        elif isinstance(size, (np.ndarray, torch.Tensor)):
            size = tuple(cp.maths.round_down_to_closest_int(num=size).tolist())
        else:
            raise TypeError('size')

        w, h = size
        if w != cap.get(3):
            cap.set(3, w)
        if h != cap.get(4):
            cap.set(4, h)

    size = w, h = cp.maths.round_down_to_closest_int(num=(cap.get(3), cap.get(4)))

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    elif isinstance(fps, (int, float)):
        pass
    else:
        raise TypeError('fps')

    spf = 1 / fps
    mspf = math.floor(spf * 1000)

    if timeout is None:
        timeout = math.inf

    if quit_key is None:
        pass
    elif isinstance(quit_key, (str, int)):
        pass
    else:
        raise TypeError('quit_key')

    if write_out_video:
        if format_out_video == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            raise ValueError('format_out_video')

        video_writer = cv2.VideoWriter(dir_out_video, fourcc, fps, (w, h), 1)
    else:
        video_writer = None

    timer = None
    t = 0

    capturing = True

    while capturing:

        if keyboard.is_pressed(quit_key):
            print('The Quitting Key was pressed. Quitting ...')
            capturing = False

        elif (t * spf) > timeout:
            print('Timeout')
            capturing = False

        # elif timer.get_seconds_total() >= timeout:
        #     print('Timeout')
        #     capturing = False
        else:

            success, image = cap.read()
            if success:

                if write_out_video:
                    video_writer.write(image)

                if isinstance(source, int) or show:
                    if timer is None:
                        timer = cp.clock.Timer(ticks_per_sec=fps)
                    else:
                        timer.wait()

                if show:
                    cv2.imshow(winname='Image', mat=image)
                    cv2.pollKey()

                t += 1

            else:
                print('Can\'t receive frame (stream end?). Exiting ...')
                capturing = False

    cv2.destroyAllWindows()
    cap.release()
    if write_out_video:
        video_writer.release()

    return None
