## IMPORTS
import os
import time
import json
import shutil
import csv
import torch
import cv2
import numpy as np
import itertools
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from PIL import Image
from shapely.geometry import Point
from shapely.affinity import scale, rotate
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, erosion
import optuna

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

                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py) ]
                        poly = [p for x in poly for p in x]
                        
                    elif type == "polygon":
                        px = anno["points"][0:-1:2]  #0 -1 2
                        py = anno["points"][1:-1:2] # 1 -1 2
                        px.append(anno["points"][0])    # 0
                        py.append(anno["points"][-1])   # -1

                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py) ]
                        poly = [p for x in poly for p in x]
                    
                    if "throat" in categoryName :
                        category_id = 0
                    elif "pore" in categoryName :
                        category_id = 1
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
def custom_mapper(dataset_dict):
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
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
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
  
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

## Replace default trainer, subs 2x 1D method
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

# keywords = ["Train", "Test"]
# for d in keywords:
#     #DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("dataset/multiclass/" + d, "dataset/multiclass/train/*.json"))
#     DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("/home/deamoon_uw_nn/DATASET/" + d + "/", 
#                                                                                    "/home/deamoon_uw_nn/DATASET/" + d + "/"))
#     MetadataCatalog.get("multiclass_Train").set( thing_classes=["throat","pore"])
  
# multiclass_metadata = MetadataCatalog.get("multiclass_Train").set( thing_classes=["throat","pore"])
# multiclass_test_metadata = MetadataCatalog.get("multiclass_Test").set( thing_classes=["throat","pore"])

def register_datasets(dataset_paths):
    for dataset_name, paths in dataset_paths.items():
        img_dir, label_dir = paths
        DatasetCatalog.register(dataset_name, lambda d=dataset_name: get_superannotate_dicts(img_dir, label_dir))
        MetadataCatalog.get(dataset_name).set(thing_classes=["throat", "pore"])

# Example usage:
dataset_paths = {
    "dataset1": ("/path/to/dataset1/images", "/path/to/dataset1/labels"),
    "dataset2": ("/path/to/dataset2/images", "/path/to/dataset2/labels"),
    "dataset3": ("/path/to/dataset3/images", "/path/to/dataset3/labels"),
}

register_datasets(dataset_paths)

# ## Def det2 hyperparameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO OPTUNA OPTIMIZATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("multiclass_Train",)
# cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.MAX_ITER = 1000
# cfg.SOLVER.STEPS = []
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
# cfg.MODEL.DEVICE = "cuda"

# ## Train model
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = CustomTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

def train_and_save_models(datasets, output_dir):
    for dataset_name in datasets:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 8
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1000
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Assuming 2 classes (throat, pore)
        cfg.MODEL.DEVICE = "cuda"
        
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        cfg.OUTPUT_DIR = dataset_output_dir
        
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # Save the model
        model_path = os.path.join(dataset_output_dir, "model_final.pth")
        torch.save(trainer.model.state_dict(), model_path)
        print(f"Model trained on {dataset_name} saved to {model_path}")

# Example usage:
train_and_save_models(dataset_paths.keys(), "./trained_models")
