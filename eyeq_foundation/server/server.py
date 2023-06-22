from fastapi import FastAPI
from eyeq_foundation import Sam
import base64
import io
from PIL import Image
import json
import numpy as np
import supervision as sv
from pydantic import BaseModel


app = FastAPI()

# Initialize Model Here:
model = Sam()


def to_json(detection: sv.Detections) -> str:
    """
    :param detection: detection as supervision format
    :return: detections as json encoded string
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    # TODO: Work on polygons
    polygons = list()
    for mask in detection.mask:
        _polygon = sv.mask_to_polygons(mask)
        polygons.append(_polygon)

    detection_dict = {'xyxy': detection.xyxy, 'class_id': detection.class_id,
                      'tracker_id': detection.tracker_id, 'confidence': detection.confidence, 'polygons': polygons}
    return json.dumps(detection_dict, cls=NumpyEncoder)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class InferData(BaseModel):
    data: str
    model_id: str


@app.post("/detect")
def detect(data: InferData):
    data = data.dict()
    model_id = data['model_id']
    buf = io.BytesIO(base64.b64decode(data['data']))
    image = Image.open(buf)
    image = np.array(image)
    image = image[:, :, ::-1].copy()

    yolov8_result = model.predict(image, imgsz=320)[0]

    detections = sv.Detections.from_yolov8(yolov8_result)


    return {"status": True, "detections": to_json(detections)}

