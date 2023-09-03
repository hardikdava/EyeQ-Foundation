# import the required library
import cv2

import supervision as sv
from eyeq_foundation import Sam, SamGuide

model = Sam(weights_path="../data/sam_vit_b_01ec64.pth", model_type="vit_b", device="cpu", guide=True)
mask_annotator = sv.MaskAnnotator()


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        sv_pt = sv.Point(x=x, y=y)
        sam_guide = SamGuide(points=sv_pt)
        detections = model.predict(image=img, input_guides=sam_guide)
        img = mask_annotator.annotate(scene=img, detections=detections)


img = cv2.imread('image0.jpg')

# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image
while True:
    cv2.imshow('Point Coordinates', img )
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()