from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import supervision as sv
from eyeq_foundation.models.sam.utils import SamGuide


class Sam:

    def __init__(self, model_type: str, weights_path: str, device: str, guide: str):
        self.predictor = None
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = device
        self._build(guide=guide)

    def _build(self, guide: str) -> None:
        sam = sam_model_registry[self.model_type](checkpoint=self.weights_path)
        if guide == "auto":
            self.predictor = SamAutomaticMaskGenerator(sam)
        elif guide == "guided":
            self.predictor = SamPredictor(sam)

    def predict(self, image: np.ndarray, input_guides: SamGuide=None) -> sv.Detections:
        detections = sv.Detections.empty()
        if isinstance(self.predictor, SamAutomaticMaskGenerator):
            sam_result = self.predictor.generate(image)
            detections = sv.Detections.from_sam(sam_result=sam_result)
        elif isinstance(self.predictor, SamPredictor):
            self.predictor.set_image(image)
            _detections = []

            for i in range(input_guides.max_input):
                xyxy = input_guides.detections.xyxy[i]
                class_id = input_guides.detections.class_id[i]

                masks, scores, logits = self.__predict_from_guide(box=xyxy, label=np.array([class_id]))

                class_id_list = np.full(shape=(masks.shape[0]), fill_value=class_id, dtype=int)
                per_box_detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks),
                    mask=masks,
                    class_id=class_id_list.astype(int)
                )
                per_box_detections = per_box_detections[per_box_detections.area == np.max(per_box_detections.area)]
                _detections.append(per_box_detections)
            detections = sv.Detections.merge(detections_list=_detections)

        return detections

    def __predict_from_guide(self, box: dict, label: np.ndarray):
        masks, scores, logits = self.predictor.predict(
            box=box,
            point_coords=None,
            point_labels=label,
            multimask_output=False
        )
        return masks, scores, logits

    @staticmethod
    def xyxy_to_xywh(xyxy):
        xyxy[2] = xyxy[2] - xyxy[0]
        xyxy[3] = xyxy[3] - xyxy[1]
        return xyxy

