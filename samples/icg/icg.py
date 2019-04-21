"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from pre-trained COCO weights
	python3 icg.py train --dataset=/path/to/icg/dataset --weights=coco

	# Resume training a model that you had trained earlier
	python3 icg.py train --dataset=/path/to/icg/dataset --weights=last

	# Train a new model starting from ImageNet weights
	python3 icg.py train --dataset=/path/to/icg/dataset --weights=imagenet

	# evaluate
	python3 icg.py val --weights=/path/to/weights/file.h5 --image=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage import filters

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

RGB_classes = {(128, 64, 128): 'paved-area',
			   (48, 41, 30): 'rocks',
			   (0, 50, 89): 'pool',
			   (28, 42, 168): 'water',
			   (107, 142, 35): 'vegetation',
			   (70, 70, 70): 'roof',
			   (102, 102, 156): 'wall',
			   (190, 153, 153): 'fence',
			   (9, 143, 150): 'car',
			   (51, 51, 0): 'tree',
			   (2, 135, 115): 'obstacle',
			   (0, 102, 0): 'grass' }


############################################################
#  Configurations
############################################################


class IcgConfig(Config):
	NAME = "icg"

	GPU_COUNT = 1

	IMAGES_PER_GPU = 1

	# Number of classes (including background)
	NUM_CLASSES = 1 + len(RGB_classes)

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 100

	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class IcgDataset(utils.Dataset):

	def load_icg(self, dataset_dir, subset):

		assert subset in ["train", "val"]
		images_dir = os.path.join(dataset_dir, 'images', subset)

		files = []
		for (dirpath, dirnames, filenames) in os.walk(images_dir):
			files.extend(filenames)

		# Add images
		for f in files:
			img_id = str(f[:-4])
			self.add_image(
				"icg", image_id=img_id,
				path=os.path.join(images_dir, img_id + '.jpg'),
				mask_image_path=os.path.join(dataset_dir, 'images_seg', img_id +'.png'))
		
		# Add classes
		index = 0
		for class_val in RGB_classes.values():
			self.add_class("icg", index, class_val)
			index += 1


	def load_mask(self, image_id):
		"""Generate instance masks for an image.
	   Returns:
		masks: A bool array of shape [height, width, instance count] with
			one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		# If not a icg dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "icg":
			return super(self.__class__, self).load_mask(image_id)

		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		info = self.image_info[image_id]
		mask_img_path = info['mask_image_path']
		color_image = imread(mask_img_path)
		gray_image = rgb2gray(color_image)
		threshold_value = filters.threshold_otsu(gray_image)
		labeled_foreground = (gray_image > threshold_value).astype(int)
		properties = regionprops(labeled_foreground, gray_image)
		height, width = color_image.shape[:2]
		mask = np.zeros([height, width, len(properties)], dtype=np.uint8)
		instance=0
		instance_masks = []
		class_ids = []
		for region in properties:
			centroid = np.rint(region.centroid).astype(np.int32)
			RGB_key = tuple(color_image[centroid[0], centroid[1]])
			print(RGB_key)
			if RGB_key not in RGB_classes:
				continue # we don't care about this label
			class_ids.append(RGB_classes[RGB_key])
			mask[region.coords[:,0], region.coords[:,1], instance] = 1
			instance += 1

		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		print(class_ids)
		return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "icg":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)


def train(model):
	"""Train the model."""
	# Training dataset.
	dataset_train = IcgDataset()
	dataset_train.load_icg(args.dataset, "train")
	dataset_train.prepare()

		
	# Validation dataset
	dataset_val = IcgDataset()
	dataset_val.load_icg(args.dataset, "val")
	dataset_val.prepare()


	# *** This training schedule is an example. Update to your needs ***
	# Since we're using a very small dataset, and starting from
	# COCO trained weights, we don't need to train too long. Also,
	# no need to train all layers, just the heads should do it.
	print("Training network heads")
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=30,
				layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN to detect objects in icg dataset.')
	parser.add_argument("command",
						metavar="<command>",
						help="'train' or 'splash'")
	parser.add_argument('--dataset', required=False,
						metavar="/path/to/icg/dataset/",
						help='Directory of the Icg dataset')
	parser.add_argument('--weights', required=True,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="/path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')
	'''
	parser.add_argument('--image', required=False,
						metavar="path or URL to image",
						help='Image to perform object detection on')
	'''
	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "splash":
		assert args.image or args.video,\
			   "Provide --image or --video to apply color splash"

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)

	# Configurations
	if args.command == "train":
		config = IcgConfig()
	else:
		class InferenceConfig(IcgConfig):
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
		config = InferenceConfig()
	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
								  model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,
								  model_dir=args.logs)

	# Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(weights_path, by_name=True, exclude=[
			"mrcnn_class_logits", "mrcnn_bbox_fc",
			"mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	# Train or evaluate
	if args.command == "train":
		train(model)
	elif args.command == "evaluate":
		pass
