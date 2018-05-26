import cv2
import numpy as np

float_formatter = lambda x: "%.2f" % x
font = cv2.FONT_HERSHEY_SIMPLEX

def drawBox(img, objects, scores, bboxes):
	boxes = []
	for idx, obj in enumerate(objects):
		score = float_formatter(scores[idx])
		bbox = bboxes[idx]

		height = img.shape[0]
		width = img.shape[1]

		ymin = int(bbox[0] * height)
		xmin = int(bbox[1] * width)
		ymax = int(bbox[2] * height)
		xmax = int(bbox[3] * width)
		boxes.append((long(xmin),long(ymin),long(xmax),long(ymax)))

		print obj, score, xmin, ymin, xmax, ymax
		
		cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
		cv2.putText(img, '{}|{}'.format(obj, score), (xmin, ymin), font, 1, (0, 128, 255), 2, cv2.LINE_AA, False)
	return boxes
		