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


# def split_dataset(img_dir, dataset_name, test_size=0.2, seed=42):
#     """
#     Splits the dataset into training and testing sets and saves the split information.
    
#     Parameters:
#     - img_dir: Directory containing images.
#     - label_dir: Directory containing labels.
#     - dataset_name: Name of the dataset.
#     - test_size: Proportion of the dataset to include in the test split.
#     - seed: Random seed for reproducibility.
    
#     Returns:
#     - train_files: List of training label files.
#     - test_files: List of testing label files.
#     """
#     random.seed(seed)
#     label_files = [f for f in os.listdir(img_dir) if f.endswith('.json')]
#     train_files, test_files = train_test_split(label_files, test_size=0.2, seed=42)

#     # Save the split
#     split_dir = "./split_dir/"
#     os.makedirs(split_dir, exist_ok=True)
#     split_file = os.path.join(split_dir, f"{dataset_name}_split.json")
#     split_data = {'train': train_files, 'test': test_files}
#     with open(split_file, 'w') as f:
#         json.dump(split_data, f)

#     print(f"Training & Testing data succesfully split into {split_file}")

#     return train_files, test_files

def register_datasets(dataset_info, test_size=0.2):
    """
    Registers the datasets in the Detectron2 framework.

    Parameters:
    - dataset_info: Dictionary containing dataset names and their info.
    - test_size: Proportion of the dataset to include in the test split.
    """
    for dataset_name, info in dataset_info.items():
        img_dir, label_dir, thing_classes = info

        # Load or split the dataset
        split_dir = "/home/deamoon_uw_nn/split_dir/"
        split_file = os.path.join(split_dir, f"{dataset_name}_split.json")
        category_json = "/home/deamoon_uw_nn/uw-com-vision/dataset_info.json"
        category_key = dataset_name
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            train_files = split_data['train']
            test_files = split_data['test']
        else:
            # train_files, test_files = split_dataset(img_dir, dataset_name, test_size=0.2)
            print(f"No split found at {split_file}")

        # Register training dataset
        DatasetCatalog.register(
            f"{dataset_name}_train",
            lambda img_dir=img_dir, label_dir=label_dir, files=train_files:
            get_split_dicts(img_dir, label_dir, files, category_json, category_key)
        )
        MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

        # Register testing dataset
        DatasetCatalog.register(
            f"{dataset_name}_test",
            lambda img_dir=img_dir, label_dir=label_dir, files=test_files:
            get_split_dicts(img_dir, label_dir, files, category_json, category_key)
        )
        MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)

# def split_dataset(img_dir, label_dir, dataset_name, test_size=0.2, seed=42):
#     """
#     Splits the dataset into training and testing sets and saves the split information.
    
#     Parameters:
#     - img_dir: Directory containing images.
#     - label_dir: Directory containing labels.
#     - dataset_name: Name of the dataset.
#     - test_size: Proportion of the dataset to include in the test split.
#     - seed: Random seed for reproducibility.
    
#     Returns:
#     - train_files: List of training label files.
#     - test_files: List of testing label files.
#     """
#     random.seed(seed)
#     label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
#     train_files, test_files = train_test_split(label_files, test_size=test_size, random_state=seed)

#     # Save the split
#     split_dir = "/home/deamoon_uw_nn/split_dir/"
#     os.makedirs(split_dir, exist_ok=True)
#     split_file = os.path.join(split_dir, f"{dataset_name}_split.json")
#     split_data = {'train': train_files, 'test': test_files}
#     with open(split_file, 'w') as f:
#         json.dump(split_data, f)

#     print(split_dir)

#     return train_files, test_files

# def load_or_split_dataset(img_dir, label_dir, dataset_name, output_dir, test_size=0.2):
#     """
#     Loads the dataset splits from CSV files or creates new splits if they don't exist.

#     Parameters:
#     - img_dir: Directory containing images.
#     - label_dir: Directory containing labels.
#     - dataset_name: Name of the dataset.
#     - output_dir: Directory to save or load split CSV files.
#     - test_size: Proportion of the dataset to include in the test split.

#     Returns:
#     - train_files: List of training label files.
#     - test_files: List of testing label files.
#     """
#     train_csv_path = os.path.join(output_dir, f"{dataset_name}_train_split.csv")
#     test_csv_path = os.path.join(output_dir, f"{dataset_name}_test_split.csv")

#     if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
#         # Load splits from CSV
#         with open(train_csv_path, 'r') as train_csv:
#             reader = csv.reader(train_csv)
#             next(reader)  # Skip header
#             train_files = [row[0] for row in reader]

#         with open(test_csv_path, 'r') as test_csv:
#             reader = csv.reader(test_csv)
#             next(reader)  # Skip header
#             test_files = [row[0] for row in reader]
        
#         print(f"Loaded training split from {train_csv_path}")
#         print(f"Loaded testing split from {test_csv_path}")
#     else:
#         # Create new splits and save them
#         train_files, test_files = split_dataset(img_dir, label_dir, dataset_name, output_dir, test_size)
    
#     return train_files, test_files

# def register_datasets(dataset_info, output_dir, test_size=0.2):

#      # Available datasets
#     # dataset_info = {
#     #     "polyhipes": ("/home/deamoon_uw_nn/DATASET/polyhipes/", "/home/deamoon_uw_nn/DATASET/polyhipes/", ["throat", "pore"])
#     # }
    
#     for dataset_name, info in dataset_info.items():
#         img_dir, label_dir, thing_classes = info

#         train_csv_path = os.path.join(output_dir, f"{dataset_name}_train_split.csv")
#         test_csv_path = os.path.join(output_dir, f"{dataset_name}_test_split.csv")

#         # Load train/test files from CSV
#         if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
#             with open(train_csv_path, 'r') as train_csv:
#                 reader = csv.reader(train_csv)
#                 next(reader)  # Skip header
#                 train_files = [row[0] for row in reader]

#             with open(test_csv_path, 'r') as test_csv:
#                 reader = csv.reader(test_csv)
#                 next(reader)  # Skip header
#                 test_files = [row[0] for row in reader]
#         else:
#             train_files, test_files = load_or_split_dataset(img_dir, label_dir, dataset_name, output_dir, test_size)

#         # Register training dataset
#         DatasetCatalog.register(
#             f"{dataset_name}_train",
#             lambda img_dir=img_dir, label_dir=label_dir, files=train_files:
#             get_split_dicts(img_dir, label_dir, files)
#         )
#         MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

#         # Register testing dataset
#         DatasetCatalog.register(
#             f"{dataset_name}_test",
#             lambda img_dir=img_dir, label_dir=label_dir, files=test_files:
#             get_split_dicts(img_dir, label_dir, files)
#         )
#         MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)

# def get_split_dicts(img_dir, label_dir, files):
#     """
#     Generates a list of dictionaries for Detectron2 dataset registration.
    
#     Parameters:
#     - img_dir: Directory containing images.
#     - label_dir: Directory containing labels.
#     - files: List of label files to process.
    
#     Returns:
#     - dataset_dicts: List of dictionaries with image and annotation data.
#     """
#     dataset_dicts = []
#     idx = 0
#     for file in files:
#         json_file = os.path.join(label_dir, file)
#         with open(json_file) as f:
#             imgs_anns = json.load(f)

#         record = {}
#         filename = os.path.join(img_dir, imgs_anns["metadata"]["name"])
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = imgs_anns["metadata"]["height"]
#         record["width"] = imgs_anns["metadata"]["width"]
#         idx += 1
#         annos = imgs_anns["instances"]
#         objs = []

#         for anno in annos:
#             categoryName = anno["className"]
#             type = anno["type"]

#             if type == "ellipse":
#                 cx = anno["cx"]
#                 cy = anno["cy"]
#                 rx = anno["rx"]
#                 ry = anno["ry"]
#                 theta = anno["angle"]
#                 ellipse = ((cx, cy), (rx, ry), theta)
#                 circ = shapely.geometry.Point(ellipse[0]).buffer(1)
#                 ell = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
#                 ellr = shapely.affinity.rotate(ell, ellipse[2])
#                 px, py = ellr.exterior.coords.xy
#                 poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#                 poly = [p for x in poly for p in x]
#             elif type == "polygon":
#                 px = anno["points"][0:-1:2]
#                 py = anno["points"][1:-1:2]
#                 px.append(anno["points"][0])
#                 py.append(anno["points"][-1])
#                 poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#                 poly = [p for x in poly for p in x]

#             if "throat" in categoryName:
#                 category_id = 0
#             elif "pore" in categoryName:
#                 category_id = 1
#             else:
#                 raise ValueError("Category Name Not Found: " + categoryName)

#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": category_id,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts

def get_split_dicts(img_dir, label_dir, files, category_json, category_key):
    """
    Generates a list of dictionaries for Detectron2 dataset registration.
    
    Parameters:
    - img_dir: Directory containing images.
    - label_dir: Directory containing labels.
    - files: List of label files to process.
    - category_json: Path to the JSON file containing category information.
    - category_key: Key in JSON to select category names.
    
    Returns:
    - dataset_dicts: List of dictionaries with image and annotation data.
    """
    # Load category names and create a mapping to category IDs
    dataset_info = read_dataset_info(category_json)
    
    # Check if category_key exists
    if category_key not in dataset_info:
        raise ValueError(f"Category key '{category_key}' not found in JSON")
    
    category_names = dataset_info[category_key][2]  # Extract category names from the JSON
    category_name_to_id = {name: idx for idx, name in enumerate(category_names)}
    
    print(f"Category Mapping: {category_name_to_id}")  # Debug: print category mapping

    dataset_dicts = []
    idx = 0
    for file in files:
        json_file = os.path.join(label_dir, file)
        with open(json_file) as f:
            imgs_anns = json.load(f)

        record = {}
        filename = os.path.join(img_dir, imgs_anns["metadata"]["name"])
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = imgs_anns["metadata"]["height"]
        record["width"] = imgs_anns["metadata"]["width"]
        idx += 1
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
                circ = shapely.geometry.Point(ellipse[0]).buffer(1)
                ell = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
                ellr = shapely.affinity.rotate(ell, ellipse[2])
                px, py = ellr.exterior.coords.xy
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
            elif type == "polygon":
                px = anno["points"][0:-1:2]
                py = anno["points"][1:-1:2]
                px.append(anno["points"][0])
                py.append(anno["points"][-1])
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

            if categoryName in category_name_to_id:
                category_id = category_name_to_id[categoryName]
            else:
                raise ValueError(f"Category Name Not Found: {categoryName}")

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
    
def custom_mapper(dataset_dicts):
    """
    Custom data mapper function for Detectron2. Applies various transformations to the image and annotations.
    
    Parameters:
    - dataset_dicts: Dictionary containing image and annotation data.
    
    Returns:
    - dataset_dicts: Updated dictionary with transformed image and annotations.
    """
    dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
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

class CustomTrainer(DefaultTrainer):
    """
    Custom trainer class extending Detectron2's DefaultTrainer to use a custom data mapper.
    """
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

def read_dataset_info(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        # Convert list values back to tuples for consistency with the original data
        dataset_info = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    return dataset_info

def train_on_dataset(dataset_name, output_dir):
    """
    Trains a model on the specified dataset.

    Parameters:
    - dataset_name: Name of the dataset to train on.
    - output_dir: Directory to save the trained model.
    """

    # Example usage
    dataset_info = read_dataset_info('/home/deamoon_uw_nn/uw-com-vision/dataset_info.json')
    register_datasets(dataset_info)

    # Load configuration
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

    thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = dataset_output_dir

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    model_path = os.path.join(dataset_output_dir, "model_final.pth")
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Model trained on {dataset_name} saved to {model_path}")
