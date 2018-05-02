# # Mask R-CNN - Train on Mirror Dataset
import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.model import log
from keras.utils import plot_model

import visdrone

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    
config = visdrone.VisDroneConfig()
config.display()

iter_num = 0

# Configuration
dataset_root_path = os.path.abspath(os.path.join(ROOT_DIR, "../data"))
train_folder = dataset_root_path + "/VisDrone2018-DET-train"
val_folder = dataset_root_path + "/VisDrone2018-DET-val"
train_images_folder = train_folder + "/images"
train_anno_folder = train_folder + "/annotations"
val_images_folder = val_folder + "/images"
val_anno_folder = val_folder + "/annotations"
train_imglist = os.listdir(train_images_folder)
train_count = len(train_imglist)
val_imglist = os.listdir(val_images_folder)
val_count = len(val_imglist)
print("Train Image Count : {} \nValidation Image Count : {}".format(train_count, val_count))

# Training dataset
dataset_train = visdrone.VisDroneDataset()
dataset_train.load_VisDrone(train_count, train_images_folder, train_imglist, train_folder)
dataset_train.prepare()

# Validation dataset
dataset_val = visdrone.VisDroneDataset()
dataset_val.load_VisDrone(val_count, val_images_folder, val_imglist, val_folder)
dataset_val.prepare()

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# images = []
# images_bbx = []
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     label, count = dataset_train.load_anno(image_id)
#     images.append(image)
#     images_bbx.append(visualize.draw_bbx(image, label, count))
# visualize.display_images(images_bbx)


### Create Model  ###
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    model.load_weights(MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# ## Training

# 1. Train the head branches
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            layers='heads')

# 2. Fine tune all layers
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=2,
#             layers="all")


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_mirror.h5")
# model.keras_model.save_weights(model_path)