"""
Script for performing zero-shot object detection
"""
from typing import Tuple, List
import os.path
from pathlib import Path
import numpy as np
import supervision as sv
from groundingdino.util.inference import Model


class Dino:
    """
    Class to perform zero shot object detection on an image using Grounding Dino
    """

    def __init__(self, model_type: str, weights_path: str, device: str):
        self.predictor = None
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = device
        self._build()

    def _build(self) -> None:
        """
        load and build model if required
        """
        current_dir = str(Path(__file__).parent)
        if self.model_type == "swint":
            config_path = os.path.join(current_dir, "swint_cfg.py")
        else:
            config_path = os.path.join(current_dir, "swinb_cfg.py")

        self.predictor = Model(model_config_path=config_path, model_checkpoint_path=self.weights_path, device=self.device)

    def predict(self, image: np.ndarray, text_promt: str, threshold: float) -> Tuple[sv.Detections, List[str]]:
        detections, labels = self.predictor.predict_with_caption(image=image, caption=text_promt,
                                            box_threshold=threshold, text_threshold=threshold)
        return detections, labels

    @staticmethod
    def dino_promt_from_labelmap(labelmap: list) -> str:
        """
        :param labelmap: list of object classes as string
        :return: string of class names joined by space as dino input
        """
        promt = ' '.join(labelmap)
        return promt



