import cv2
import numpy as np
tl=[]
def drawBox(img, objects, scores, bboxes):
	for idx, obj in enumerate(objects):
		if obj =='person':
			bbox = bboxes[idx]

			height = img.shape[0]
			width = img.shape[1]

			ymin = int(bbox[0] * height)
			xmin = int(bbox[1] * width)
			ymax = int(bbox[2] * height)
			xmax = int(bbox[3] * width)

			tl.append((ymin,xmin,ymax,xmax))
	return(tl)
		
		
