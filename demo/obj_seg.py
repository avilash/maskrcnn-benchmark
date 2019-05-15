import argparse
import cv2
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time

from flask import Flask
from flask import request
import json

def merge_masks(masks):
	final_mask = np.zeros((masks.shape[2], masks.shape[3]), dtype=np.uint8)
	colors = []
	num_masks = masks.shape[0]
	color_delta = int(200/num_masks)
	for i, mask in enumerate(masks):
		color = 25 + i*color_delta
		colors.append(color)
		mask = mask[0]
		final_mask[mask == 1] = color
	return final_mask, colors

def warmup():
	img = cv2.imread('test.jpg')
	result, boxes, masks, labels = coco_demo.run_on_opencv_image2(img)
	cv2.imwrite("result.png", result)
	final_mask, colors = merge_masks(masks)
	cv2.imwrite("result_mask.png", final_mask)
	result = []
	for label, box, color in zip(labels, boxes, colors):
		detection = {
			'label': label.numpy().tolist(),
			'bbox' : box.numpy().tolist(),
			'color': color
		}
		result.append(detection)
	print(result)

def test(src):
	valid_ext = [".jpg"]
	import os
	img_path = os.path.join(src, 'JPEGImages')
	result_path = os.path.join(src, 'Result')
	for f in os.listdir(img_path):
		file_ext = os.path.splitext(f)[1]
		file_name = os.path.splitext(f)[0]
		if file_ext.lower() not in valid_ext:
			continue
		print (f)
		img = cv2.imread(os.path.join(img_path, f))
		result = coco_demo.run_on_opencv_image(img)
		cv2.imwrite(os.path.join(result_path, f), result)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Start a Caching Worker')
	parser.add_argument('--config_file', default='prod.yaml', type=str,
                help='Config file path')

	args = parser.parse_args()

	cfg.merge_from_file(args.config_file)
	cfg.freeze()
	coco_demo = COCODemo(
	    cfg,
	    confidence_threshold=0.7,
	    min_image_size=800,
	)

	app = Flask(__name__)

	@app.route("/object_seg", methods=['POST'])
	def detect():
		im_file = request.files['file']
		img = cv2.imdecode(np.fromstring(im_file.read(), np.uint8), cv2.IMREAD_COLOR)
		result, boxes, masks, labels = coco_demo.run_on_opencv_image2(img)
		final_mask, colors = merge_masks(masks)
		detections = []
		for label, box, color in zip(labels, boxes, colors):
			detection = {
				'label': label.numpy().tolist(),
				'bbox' : box.numpy().tolist(),
				'color': color
			}
			detections.append(detection)
		return json.dumps({'dets':detections, 'mask':final_mask.tolist()}), 200

	print("Running http server ... ")
	app.run(host='0.0.0.0', port=9494)
	# test('/home/pickpal/Work/HDD/Dataset/150OBJ/Test/Set13')
	
