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
import random
import copy
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
from sklearn.model_selection import train_test_split
from data_preparation import split_dataset
from data_preparation import split_dataset, register_datasets, get_split_dicts
from data_preparation import get_trained_model_paths, load_model, choose_and_use_model, read_dataset_info

def custom_mapper(dataset_dicts):
    """
    Custom data mapper function for Detectron2. Applies various transformations to the image and annotations.
    
    Parameters:
    - dataset_dicts: Dictionary containing image and annotation data.
    
    Returns:
    - dataset_dicts: Updated dictionary with transformed image and annotations.
    """
    dataset_dicts = copy.deepcopy(dataset_dicts)  # It will be modified by the code below
    image = utils.read_image(dataset_dicts["file_name"], format="BGR")
    
    transform_list = [
        T.Resize((800, 800)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]

    # Apply transformations
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dicts["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    # Transform annotations
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dicts.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    # Create instances from annotations
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dicts["instances"] = utils.filter_empty_instances(instances)
    
    return dataset_dicts

class CustomTrainer(DefaultTrainer):
    """
    Custom trainer class extending Detectron2's DefaultTrainer to use a custom data mapper.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

def train_on_dataset(dataset_name, output_dir):
    """
    Trains a model on the specified dataset.

    Parameters:
    - dataset_name: Name of the dataset.
    - output_dir: Directory to save the trained model.
    """
    # Read dataset information
    dataset_info = read_dataset_info('/home/deamoon_uw_nn/uw-com-vision/dataset_info.json')
    
    # Register datasets
    register_datasets({dataset_name: dataset_info[dataset_name]})

    # Debug prints for verification
    print(DatasetCatalog.get(f"{dataset_name}_train"))
    print(DatasetCatalog.get(f"{dataset_name}_test"))
    
    # Path for the split file
    split_file = os.path.join("/home/deamoon_uw_nn/split_dir/", f"{dataset_name}_split.json")
    print(f"Split file for {dataset_name}: {split_file}")

    # Configuration for training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
    cfg.DATASETS.TEST = (f"{dataset_name}_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32

    # Set the number of classes
    thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directory for the dataset
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = dataset_output_dir

    # Initialize and start the trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save the trained model
    model_path = os.path.join(dataset_output_dir, "model_final.pth")
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Model trained on {dataset_name} saved to {model_path}")
