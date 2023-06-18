import cv2
from eyeq_foundation import Annotator, visualizer, Dino


labelmaps = ["person", "tie"]

img = cv2.imread("../data/zidane.jpg")

text_promt = Dino.dino_promt_from_labelmap(labelmap=labelmaps)

dino_model = Dino( model_type="swint", weights_path="../weights/groundingdino_swint_ogc.pth", device="cpu")
detections_dino, labels = dino_model.predict(image=img, text_promt=text_promt, threshold=0.4)

annotator = Annotator()
render_img = annotator.annotate(img, detections=detections_dino, labels=labels)
visualizer("Result", render_img, wait_key=0)
