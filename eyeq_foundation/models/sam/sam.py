from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import supervision as sv
from typing import List


class SamGuide:

    boxes: sv.Detections = sv.Detections.empty()
    points: sv.Point = None

    def __init__(self, detections: sv.Detections = sv.Detections.empty(), points: sv.Point = None):
        self.detections = detections
        self.points = points
    # @classmethod
    # def from_numpy_points(cls, points: np.ndarray):
    #     if points.shape[0] > 0:
    #         for pt in points:
    #             sv_pt = sv.Point(x=pt[0], y=pt[1])
    #             points_guides.append(sv_pt)
    #     return cls(detections=sv.Detections.empty(), points=points_guides)

    # @classmethod
    # def from_numpy_boxes(cls, boxes: np.ndarray):
    #     detections = sv.Detections.empty()
    #     if boxes.shape[0] > 0:
    #         detections = sv.Detections(xyxy=boxes)
    #     return cls(detections=detections, points=[])


class Sam:

    def __init__(self, model_type: str, weights_path: str, device: str, guide: bool):
        self.predictor = None
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = device
        self.guide = guide
        self._build()

    def _build(self) -> None:
        sam = sam_model_registry[self.model_type](checkpoint=self.weights_path)
        if self.guide:
            self.predictor = SamPredictor(sam)
        else:
            self.predictor = SamAutomaticMaskGenerator(sam)

    def predict(self, image: np.ndarray, input_guides: SamGuide=None) -> sv.Detections:
        detections = sv.Detections.empty()
        if self.guide:
            self.predictor.set_image(image)
            _detections = []

            if input_guides.points:
                pt = input_guides.points
                masks, scores, logits = self._predict_from_points(points=np.array([[pt.x, pt.y]]))
                detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)

        else:
            sam_result = self.predictor.generate(image)
            detections = sv.Detections.from_sam(sam_result=sam_result)

        return detections

            # for i in range(SamGuide.boxes.xyxy):
            #     xyxy = input_guides.detections.xyxy[i]
            #     class_id = input_guides.detections.class_id[i]
            #
            #     masks, scores, logits = self.__predict_from_guide(box=xyxy, label=np.array([class_id]))
            #
            #     class_id_list = np.full(shape=(masks.shape[0]), fill_value=class_id, dtype=int)
            #     per_box_detections = sv.Detections(
            #         xyxy=sv.mask_to_xyxy(masks=masks),
            #         mask=masks,
            #         class_id=class_id_list.astype(int)
            #     )
            #     per_box_detections = per_box_detections[per_box_detections.area == np.max(per_box_detections.area)]
            #     _detections.append(per_box_detections)
            # detections = sv.Detections.merge(detections_list=_detections)
        # else:
        #     sam_result = self.predictor.generate(image)
        #     detections = sv.Detections.from_sam(sam_result=sam_result)
        #
        # return detections

    def _predict_from_boxes(self, box: np.ndarray, label: np.ndarray):
        masks, scores, logits = self.predictor.predict(
            box=box,
            point_coords=None,
            point_labels=label,
            multimask_output=False
        )
        return masks, scores, logits

    def _predict_from_points(self, points: np.ndarray, labels: np.array = None):
        if not labels:
            labels = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        return masks, scores, logits

    @staticmethod
    def xyxy_to_xywh(xyxy):
        xyxy[2] = xyxy[2] - xyxy[0]
        xyxy[3] = xyxy[3] - xyxy[1]
        return xyxy



