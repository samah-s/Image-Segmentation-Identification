import cv2
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt

# Import Mask RCNN
sys.path.append(os.path.join(os.getcwd(), "Mask_RCNN"))
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import log

# Configuration class
class InferenceConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + background class

# Load the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(),
                          model_dir=os.getcwd())
model.load_weights("mask_rcnn_coco.h5", by_name=True)

# Load an image
image = cv2.imread("input_image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
results = model.detect([image_rgb], verbose=1)

# Visualize the results
r = results[0]
visualize.display_instances(image_rgb, r['rois'], r['masks'], r['class_ids'], 
                            ['BG'] + list(range(1, 81)), r['scores'])

