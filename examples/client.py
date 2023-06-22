import json
import numpy as np
import requests
import cv2
import base64
import supervision as sv
from eyeq_foundation import Annotator, visualizer


HOST = "127.0.0.1"
PORT = 8000


def to_dict(json_response) -> sv.Detections:
    detections_dict = json.loads(json_response["detections"])

    def get_value(key: str):
        querry_value = detections_dict.get(key, None)
        return np.asarray(querry_value) if querry_value else None

    detection = sv.Detections(xyxy=get_value('xyxy'),
                              class_id=get_value('class_id'),
                              confidence=get_value('confidence'),
                              tracker_id=get_value('tracker_id'))
    return detection


path_img = "../data/zidane.jpg"

url = f'http://{HOST}:{PORT}/detect'

# Read Image
with open(path_img, 'rb') as image_string:
    byte_string = base64.b64encode(image_string.read()).decode('utf-8')

res = requests.post(url, json={'data': byte_string, 'model_id': 'yolov8s.pt'})

# Read Image as cv for visualization purpose
img = cv2.imread(path_img)

detection_json = res.json()

detection = to_dict(detection_json)

annotator = Annotator()
render_img = annotator.annotate(image=img, detections=detection)

visualizer("Result", render_img, 0)



