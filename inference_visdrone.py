# # Mask R-CNN - Test on VisDrone Validation Dataset
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import mrcnn.utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
VISDRONE_MODEL_PATH = os.path.join(MODEL_DIR, "visdrone20180504T2326", "mask_rcnn_visdrone_0014.h5")

# Directory of images to run detection on
DATASET_ROOT_PATH = os.path.abspath(os.path.join(ROOT_DIR, "../data"))
IMAGE_DIR = os.path.join(DATASET_ROOT_PATH + "/VisDrone2018-DET-test-challenge/images")
OUTPUT_PATH = os.path.join(DATASET_ROOT_PATH + "VisDrone2018-DET-test-challenge/outputs")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

## Configurations
class InferenceConfig(Config):
    """Derives from the base Config class and overrides values specific to the Mirror dataset"""
    NAME = "VisDrone"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2 + 10  # Mirror has only one (mirror) class
    DETECTION_MIN_CONFIDENCE = 0.9


config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained by TaylorMei
model.load_weights(VISDRONE_MODEL_PATH, by_name=True)


# VISDRONE Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['ignored', 'pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricyvle', 'awning-tricycle', 'bus', 'motor', 'others']


# ## Run Object Detection
imglist = os.listdir(IMAGE_DIR)
print(len(imglist))
for imgname in imglist:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, imgname))
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results
    print(results[0])
    r = results[0]
    print(r['rois'], r['class_ids'], r['scores'])

    # label_txt = open(OUTPUT_PATH + "/" + imgname + ".txt", "rw")
    # try:
    #     count = 0
    #     for line in label_txt:
    #         (x, y, w, h, score, cls, truncation, occlusion) = line.split(',')
    #         label.append((x, y, w, h, score, cls, truncation, occlusion))
    #         count += 1
    # finally:
    #     label_txt.close()


