## IMPORTS
import os
from os import listdir
import time
import glob
import random
import copy
import json
import datetime
from datetime import timedelta
import shutil
from distutils import file_util, dir_util
from distutils.dir_util import copy_tree
from contextlib import redirect_stdout
import tempfile
import statistics
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import csv
import torch
import torchvision
import cv2
import numpy as np
import itertools
from itertools import groupby
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from PIL import Image
import seaborn as sns
import shapely
from shapely.geometry import Point
from shapely.affinity import scale, rotate
import matplotlib.pyplot as plt
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import pycocotools.mask as mask_util
from skimage.measure import find_contours
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, erosion
from google.cloud import storage

## Def for dataset build, SA annotated data, SA format, WARNING, NO POLYLINES
def get_superannotate_dicts(img_dir, label_dir):
    dataset_dicts = []
    idx = 0
    for r, d, f in os.walk(label_dir):
        for file in f:
            if file.endswith(".json"):
                json_file = os.path.join(r, file)
                print(json_file)

                with open(json_file) as f:
                    imgs_anns = json.load(f)

                record = {}
                filename = os.path.join(img_dir, imgs_anns["metadata"]["name"])
                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = imgs_anns["metadata"]["height"]
                record["width"] = imgs_anns["metadata"]["width"]
                idx = idx + 1
                annos = imgs_anns["instances"]
                objs = []
                
                for anno in annos:
                    categoryName = anno["className"]
                    type = anno["type"]

                    if type == "ellipse":
                        cx = anno["cx"]
                        cy = anno["cy"]
                        rx = anno["rx"]
                        ry = anno["ry"]
                        theta = anno["angle"]
                        ellipse = ((cx, cy), (rx, ry), theta)
                        # Create a circle of radius 1 around the centre point:
                        circ = shapely.geometry.Point(ellipse[0]).buffer(1)
                        # Create ellipse along x and y:
                        ell = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
                        # rotate the ellipse(clockwise, x axis pointing right):
                        ellr = shapely.affinity.rotate(ell, ellipse[2])

                        px, py = ellr.exterior.coords.xy
                    elif type == "polygon":
                        px = anno["points"][0:-1:2]  #0 -1 2
                        py = anno["points"][1:-1:2] # 1 -1 2
                        px.append(anno["points"][0])    # 0
                        py.append(anno["points"][-1])   # -1
                      
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py) ]
                    poly = [p for x in poly for p in x]

                    if "Scale bar" in categoryName :
                        category_id = 0
                    elif "Wall thickness of polyHIPEs" in categoryName :
                        category_id = 1
                    elif "Pore throats of polyHIPEs" in categoryName :
                        category_id = 2
                    elif "Pores of polyHIPEs" in categoryName :
                        category_id = 3
                    else:
                        raise ValueError("Category Name Not Found: "+ categoryName)

                    obj = {
                        "bbox":[np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": category_id,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts

## Def custom mapper, rand changes to dataset imgs, induce variability to dataset
def custom_mapper(dataset_dicts):
    dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
    image = utils.read_image(dataset_dicts["file_name"], format="BGR")
    transform_list = [
        T.Resize((800,800)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
  
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dicts["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dicts.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
  
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dicts["instances"] = utils.filter_empty_instances(instances)
    return dataset_dicts

## Replace default trainer, subs 2x 1D method
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

## Load custom dataset
# Load detection classes  thing_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])

csv_file_path = '/home/deamoon_uw_nn/classes.csv'

# Lists to store class names and colors
det_classes = []
det_colors = []

with open(csv_file_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        det_classes.append(row['className'])
        # Assuming red, green, blue are stored as separate columns
        red = int(row['red'])
        green = int(row['green'])
        blue = int(row['blue'])
        det_colors.append((red, green, blue))


## Load custom dataset, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE THING CLASSES TO LOAD FROM FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Dataset load
keywords = ["Train", "Test"]
for d in keywords:
    #DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("dataset/multiclass/" + d, "dataset/multiclass/train/*.json"))
    DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("/home/deamoon_uw_nn/DATASET/" + d + "/", 
                                                                                   "/home/deamoon_uw_nn/DATASET/" + d + "/"))
    MetadataCatalog.get("multiclass_Train").set( thing_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])
  
multiclass_metadata = MetadataCatalog.get("multiclass_Train").set( thing_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])
multiclass_test_metadata = MetadataCatalog.get("multiclass_Test").set( thing_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])
## Def det2 hyperparameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO OPTUNA OPTIMIZATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("multiclass_Train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.DEVICE = "cuda"

## Train model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
