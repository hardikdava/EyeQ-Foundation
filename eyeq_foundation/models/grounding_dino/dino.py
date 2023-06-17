import os.path
from torchvision.ops import box_convert
import numpy as np
import cv2
import torch
import supervision as sv
from groundingdino.util.inference import load_model, predict, load_image, annotate
from pathlib import Path
import groundingdino.datasets.transforms as T


class DinoGuide:

    def __init__(self):
        pass

    @staticmethod
    def promt_from_labelmap(labelmap: list) -> str:
        promt = ' '.join(labelmap)
        return promt


class Dino:

    def __init__(self, model_type: str, weights_path: str, device: str):
        self.predictor = None
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = device
        self._build()

    def _build(self):
        current_dir = str(Path(__file__).parent)
        if self.model_type == "swint":
            config_path = os.path.join(current_dir, "swint_cfg.py")
        else:
            config_path = os.path.join(current_dir, "swinb_cfg.py")
        self.predictor = load_model(config_path,self.weights_path)

    def _preprocess(self, image: np.ndarray):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed, _ = transform(image_rgb, None)
        return image, image_transformed

    def predict(self, image_path: str, TEXT_PROMPT, BOX_TRESHOLD=0.5, TEXT_TRESHOLD=0.5):
        image_source, image = load_image(image_path)
        raw_boxes, logits, phrases = predict(
            model=self.predictor,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=self.device
        )
        print("logits:", logits)
        print("pphrases", phrases)
        h, w, _ = image_source.shape
        boxes = raw_boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        detections = sv.Detections(xyxy=xyxy)
        return detections




