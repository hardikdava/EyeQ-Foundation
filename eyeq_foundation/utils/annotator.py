import supervision as sv
import numpy as np


class Annotator:

    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

    def annotate(self, image=np.ndarray, detections=sv.Detections, labels=None, *args, **kwargs):
        image = self.box_annotator.annotate(scene=image, detections=detections, labels=labels)
        image = self.mask_annotator.annotate(scene=image, detections=detections)
        return image
