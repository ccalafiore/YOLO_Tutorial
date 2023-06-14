
import os
import utilities


root_script = __file__
root_project = os.path.dirname(root_script)
root_data = os.path.join(root_project, 'data')

root_images = os.path.join(root_data, 'images')
root_input_images = os.path.join(root_images, 'input_images')

name_input_images = [
    'bus_rapid_transit.jpg', 'people_cars_bicycles_1.jpg', 'people_cars_bicycles_2.jpg', 'dino_1.jpg',
    'amici_dispersi.jpg', 'trial.jpg', 'dinner.jpg']

I = len(name_input_images)
dir_input_images = [os.path.join(root_input_images, name_i) for name_i in name_input_images]


root_yolos = os.path.join(root_data, 'yolos')
name_yolo_versions = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
v = 4
name_yolo_version_v = name_yolo_versions[v]
dir_yolo_version_v = os.path.join(root_yolos, name_yolo_version_v)

if os.path.exists(dir_yolo_version_v):
    yolo_v = utilities.load_yolo(dir_file=dir_yolo_version_v)
else:
    yolo_v = utilities.download_yolo(name=name_yolo_version_v, dir_file=dir_yolo_version_v)

results_v = yolo_v(source=dir_input_images, verbose=False)

xyxy = [results_v[i].boxes.xyxy for i in range(0, I, 1)]
classes = [results_v[i].boxes.cls for i in range(0, I, 1)]
confidences = [results_v[i].boxes.conf for i in range(0, I, 1)]


box_drawer = utilities.BoxDrawer(names=yolo_v.model.names, colors=None, threshold=(255 * .3))

root_out_images = os.path.join(root_images, 'output_images', name_yolo_version_v.removesuffix('.pt'))
os.makedirs(name=root_out_images, exist_ok=True)
dir_out_images = [os.path.join(root_out_images, name_i) for name_i in name_input_images]

out_images = box_drawer(
    input_images=dir_input_images, xyxy=xyxy, dir_out_images=dir_out_images, n_workers=None,
    classes=classes, confidences=confidences)
