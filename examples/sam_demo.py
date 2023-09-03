import cv2
import supervision as sv
from eyeq_foundation import Sam, SamGuide


model = Sam(weights_path="../data/sam_vit_b_01ec64.pth", model_type="vit_b", device="cpu", guide=False)
# read the input image
original_img = cv2.imread('image0.jpg')

detections = model.predict(image=original_img)
print(len(detections))

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

img = box_annotator.annotate(original_img, detections)
img = mask_annotator.annotate(original_img, detections)

cv2.imshow("SAM", img)
cv2.waitKey(0)













