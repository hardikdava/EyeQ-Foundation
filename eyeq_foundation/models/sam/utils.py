import numpy as np
import supervision as sv


class SamGuide:

    def __init__(self, detections: sv.Detections = sv.Detections.empty(), points: np.ndarray = None):
        self.detections = detections
        self.points = points
        self.max_input = detections.xyxy.shape[0]

    @staticmethod
    def promt_from_labelmap(labelmap: list) -> str:
        promt = ' '.join(labelmap)
        return promt




