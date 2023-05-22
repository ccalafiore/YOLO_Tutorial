
# todo

import os
import utilities
import calapy as cp
import cv2


root_script = __file__
root_project = os.path.dirname(root_script)
root_data = os.path.join(root_project, 'data')

root_yolos = os.path.join(root_data, 'yolos')
name_yolo_versions = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
v = 1
name_yolo_version_v = name_yolo_versions[v]
dir_yolo_version_v = os.path.join(root_yolos, name_yolo_version_v)

yolo_v = utilities.load_yolo(name=name_yolo_version_v, dir_file=dir_yolo_version_v)

box_drawer = utilities.BoxDrawer(names=yolo_v.model.names, colors=None, threshold=(255 * .5))

camera_index = 1

quit_key_str = 'q'
quit_key_int = ord(quit_key_str)

timeout = 20




cap = cv2.VideoCapture(index=camera_index)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(3, 1289)
cap.set(4, 720)

# start the model by processing one image
success, image = cap.read()
results_v = yolo_v(source=image, stream=False, verbose=False)

playing = True

timer = cp.clock.Timer()

while playing:

    success, image = cap.read()

    if success:

        results_v = yolo_v(source=image, stream=True, verbose=False)
        for results_vi in results_v:
            image = box_drawer(boxes=results_vi.boxes, input_image=image, dir_out_image=None)

        # results_v = yolo_v(source=image, stream=False, verbose=False)
        # image = box_drawer(boxes=results_v[0].boxes, input_image=image, dir_out_image=None)

        cv2.imshow(winname='Image', mat=image)

        if (cv2.waitKey(delay=1) == quit_key_int) or (timer.get_seconds() >= timeout):
            playing = False
    else:
        print("Can't receive frame (stream end?). Exiting ...")
        playing = False

cap.release()
cv2.destroyAllWindows()
