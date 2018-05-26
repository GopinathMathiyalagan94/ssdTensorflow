import cv2
import skvideo.io
import skvideo.measure
import numpy as np

outputfile = "test.mp4"
# start the FFmpeg writing subprocess with following parameters
writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
  '-vcodec': 'libx264', '-b': '300000000'
})



url = "rtsp://admin:admin123@192.168.31.202:554/profile2"
cam = cv2.VideoCapture(url)

while True:
	ret, img = cam.read()
	if not ret: continue

	writer.writeFrame(img)

	cv2.imshow("202", img)
	if cv2.waitKey(1) & 0xFF == 27: break

writer.close()