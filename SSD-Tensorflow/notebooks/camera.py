# import cv2

# url = "rtsp://admin:admin123@192.168.31.202:554/profile1"
# cam = cv2.VideoCapture(url)

# while True:
# 	ret, img = cam.read()
# 	if not ret: continue

# 	cv2.imshow("cam-202", img)
# 	# skipping frames
# 	for i in range(0, 10):
# 		ret, img = cam.read()

# 	if cv2.waitKey(10) == 27: break

# cam.release()

import os
path = '/home/jarvis/Face-Input-Videos'
vid_names = sorted(os.listdir(path))
print vid_names
