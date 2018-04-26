import numpy as np
from PIL import Image
from mrcnn.config import Config
import mrcnn.utils as utils
import yaml


### Configurations
class DroneConfig(Config):
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
    NUM_CLASSES = 1 + 10  # background + 10 category

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    # Actually scale is square root of RPN's area
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 10

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # skip detection with <x% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



### Dataset
class DroneDataset(utils.Dataset):

    def get_obj_index(self, image):
        """Get the number of instance in the image
        """
        n = np.max(image)
        return n

    def from_yaml_get_class(self,image_id):
        """Translate the yaml file to get label """
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            """j is row and i is column"""
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index +1:
                        mask[j, i, index] = 1
        return mask

    def load_VisDrone(self, count, height, width, img_folder, mask_folder,
                    imglist, dataset_root_path):
        self.add_class("VisDrone", 1, "pedestrian")
        self.add_class("VisDrone", 2, "person")
        self.add_class("VisDrone", 3, "car")
        self.add_class("VisDrone", 4, "van")
        self.add_class("VisDrone", 5, "bus")
        self.add_class("VisDrone", 6, "truck")
        self.add_class("VisDrone", 7, "motor")
        self.add_class("VisDrone", 8, "bicycle")
        self.add_class("VisDrone", 9, "awning-tricycle")
        self.add_class("VisDrone", 10, "tricycle")
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            # filestr = filestr.split("_")[1]
            mask_path = mask_folder + "/" + filestr + "_json/label8.png"
            yaml_path = mask_folder + "/" + filestr + "_json/info.yaml"
            self.add_image("shapes", image_id=i, path=img_folder + "/" + imglist[i],
                           width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)


    def load_mask(self, image_id):
        global iter_num
        info = self.image_info[image_id]
        count = 1
        img = Image.open(info['mask_path'])
        width, height = img.size
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        labels_form=[]
        for i in range(len(labels)):
            if labels[i].find("mirror")!=-1:
                #print "box"
                labels_form.append("mirror")
            # elif labels[i].find("column")!=-1:
            #     #print "column"
            #     labels_form.append("column")
            # elif labels[i].find("package")!=-1:
            #     #print "package"
            #     labels_form.append("package")
            # elif labels[i].find("fruit")!=-1:
            #     #print "fruit"
            #     labels_form.append("fruit")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)



