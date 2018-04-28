import numpy as np
from PIL import Image
from mrcnn.config import Config
import mrcnn.utils as utils

### Configurations
class VisDroneConfig(Config):
    """Configuration for training on the mirror dataset.
    Derives from the base Config class and overrides values specific
    to the mirror dataset.
    """
    # Give the configuration a recognizable name
    NAME = "VisDrone"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2 + 10  # background + 10 category

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    # Actually scale is square root of RPN's area
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 6471

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 100

    # skip detection with <x% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


### Dataset
class VisDroneDataset(utils.Dataset):

    # TaylorMei
    def load_VisDrone(self, count, img_folder, imglist, dataset_path):
        self.add_class("VisDrone", 1, "pedestrian")
        self.add_class("VisDrone", 2, "people")
        self.add_class("VisDrone", 3, "bicycle")
        self.add_class("VisDrone", 4, "car")
        self.add_class("VisDrone", 5, "van")
        self.add_class("VisDrone", 6, "truck")
        self.add_class("VisDrone", 7, "tricycle")
        self.add_class("VisDrone", 8, "awning-tricycle")
        self.add_class("VisDrone", 9, "bus")
        self.add_class("VisDrone", 10, "motor")
        for i in range(count):
            image_name = imglist[i].split(".")[0]
            anno_path = dataset_path + "/annotations/" + image_name + ".txt"
            self.add_image("VisDrone", image_id=i, path=img_folder + "/" + imglist[i],
                        anno_path=anno_path)

    # TaylorMei
    def load_anno(self, image_id):

        label = []
        info = self.image_info[image_id]
        label_txt = open(info["anno_path"], "r")
        try:
            count = 0
            for line in label_txt:
                (x, y, w, h, score, cls, truncation, occlusion) = line.split(',')
                label.append((x, y, w, h, score, cls, truncation, occlusion))
                count += 1
        finally:
            label_txt.close()

        return label, count



