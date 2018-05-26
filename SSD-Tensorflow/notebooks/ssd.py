import os
import time
import math
import random
import dlib

import numpy as np
import tensorflow as tf
import cv2

import skvideo.io
import skvideo.measure
import numpy as np

slim = tf.contrib.slim

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from class_mapping import getClassName, getClassCategory
from draw import drawBox
# from testdraw import drawBox
# from notebooks import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.2, nms_threshold=.15, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
# path = '../demo/'
# image_names = sorted(os.listdir(path))
# img = mpimg.imread(path + image_names[-5])
def exact_points(img,points):
	out=[]
	for point in points:
        #to find min and max x coordinates
		height = img.shape[0]
		width = img.shape[1]

		if point[0]<point[2]:
			minx=point[0]* height
			maxx=point[2]* height
		else:
			minx=point[2]* height
			maxx=point[0]* height
        #to find min and max y coordinates
		if point[1]<point[3]:
			miny=point[1]* width
			maxy=point[3]* width
		else:
			miny=point[3]* width
			maxy=point[1]* width
		out.append((long(minx),long(miny),long(maxx),long(maxy)))
		cv2.rectangle(img, (int(minx), int(miny)), (int(maxx), int(maxy)), (0, 0, 255), 2)

	return out


def create_tracker(a):
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
	tracker_type = tracker_types[a]
	print(major_ver,minor_ver,subminor_ver)
	if int(minor_ver) < 3:
		tracker = cv2.Tracker_create(tracker_type)
	else:
		if tracker_type == 'BOOSTING':
			tracker = cv2.TrackerBoosting_create()
		if tracker_type == 'MIL':
			tracker = cv2.TrackerMIL_create()
		if tracker_type == 'KCF':
			tracker = cv2.TrackerKCF_create()
		if tracker_type == 'TLD':
			tracker = cv2.TrackerTLD_create()
		if tracker_type == 'MEDIANFLOW':
			tracker = cv2.TrackerMedianFlow_create()
		if tracker_type == 'GOTURN':
			tracker = cv2.TrackerGOTURN_create()

	return(tracker)


path = '/home/jarvis/'
vid_names = sorted(os.listdir(path))
# vid_names = ["Samudra-C3-5ad40d5872fe9f14f456a3a9-Winning.mp4", "Samudra-C3-5ad3463772fe9f14a36a3bc4.mp4"]
vid_names = ["vehicle.mp4"]
print len(vid_names)

url = "rtsp://admin:admin@192.168.31.108:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"
for vid_name in vid_names:
	# url = path + "" + vid_name
	print url
	cam = cv2.VideoCapture(url)

	# outputfile = vid_name
	# start the FFmpeg writing subprocess with following parameters
	# writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
	# 	'-vcodec': 'libx264', '-b': '300000000'
	# 	})

	# while True:
	# 	ret, img = cam.read()
	# 	if not ret: break
	# 	dimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
	# 	rclasses, rscores, rbboxes =  process_image(dimg)


	# 	t = [getClassName(i) for i in rclasses]
	# 	print(t)
	# 	drawBox(img, t, rscores, rbboxes)

	# 	writer.writeFrame(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
	# 	cv2.imshow(vid_name, img)
	# 	if cv2.waitKey(10) == 27: break

	# writer.close()
	# cam.release()

	frame_cnt=0
	tracker = []

	while True:
		ret, img = cam.read()
		if not ret: break
		
		if True: #frame_cnt%10 == 0:	
			start = time.time()
			dimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
			rclasses, rscores, rbboxes =  process_image(dimg)
			print 'Detection + Display took %s'%(time.time()-start)
			print rclasses
			t = [getClassName(i) for i in rclasses]
			print(t)
			bbox = drawBox(img, t, rscores, rbboxes)

			# tracker = [dlib.correlation_tracker() for _ in xrange(len(rbboxes))]
			# trkr = [tracker[i].start_track(img, dlib.rectangle(*rect)) for i, rect in enumerate(bbox)]

			# for rect in bbox:
			# 	_trkr = dlib.correlation_tracker()
			# 	_trkr.start_track(img, dlib.rectangle(rect[0], rect[1], rect[2], rect[3]))
			# 	tracker.append(_trkr)

			# print(tracker)

		# else:
		# 	print('Total tracker ', len(tracker))
		# 	for i in xrange(len(tracker)):
		# 		tracker[i].update(img)
		# 		# Get the position of th object, draw a 
		# 		# bounding box around it and display it.
		# 		rect = tracker[i].get_position()
		# 		pt1 = (int(rect.left()), int(rect.top()))
		# 		pt2 = (int(rect.right()), int(rect.bottom()))
		# 		cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
		# 		print "Object {} tracked at [{}, {}] \r".format(i, pt1, pt2)

				# cv2.imshow("Image", img)
    		    # Continue until the user presses ESC key
				# if cv2.waitKey(1) == 27: break

				# ok, bbox = tracker.update(img)
				# img = cv2.pyrDown(img)

				# if ok:
				# 	p1 = (int(bbox[0]), int(bbox[1]))
				# 	p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				# 	cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)



				# print rclasses
				# print 'Detection took %s'%(time.time()-start)
				#t = [getClassName(i) for i in rclasses]
				# print(rclasses, t, rscores)
				# drawBox(img, t, rscores, rbboxes)
				# print 'Detection + Display took %s'%(time.time()-start)

				# skipping frames
				# for i in range(0, 3):
				# 	ret, img = cam.read()
		# img = cv2.pyrDown(img)
		cv2.imshow("cam-202", img)
		if cv2.waitKey(10) == 27: break

		frame_cnt+=1

	# writer.close()
	cam.release()

		