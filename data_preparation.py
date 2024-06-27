import os
import json
import csv
import random
from sklearn.model_selection import train_test_split
import shapely.geometry
import shapely.affinity
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Constant paths
SPLIT_DIR = "/home/deamoon_uw_nn/split_dir/"
CATEGORY_JSON = "/home/deamoon_uw_nn/uw-com-vision/dataset_info.json"

def split_dataset(img_dir, dataset_name, test_size=0.2, seed=42):
    """
    Splits the dataset into training and testing sets and saves the split information.
    
    Parameters:
    - img_dir: Directory containing images.
    - dataset_name: Name of the dataset.
    - test_size: Proportion of the dataset to include in the test split.
    - seed: Random seed for reproducibility.
    
    Returns:
    - train_files: List of training label files.
    - test_files: List of testing label files.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # List all label files in the image directory
    label_files = [f for f in os.listdir(img_dir) if f.endswith('.json')]
    
    # Split the label files into training and testing sets
    train_files, test_files = train_test_split(label_files, test_size=test_size, random_state=seed)

    # Create directory to save the split information if it doesn't exist
    os.makedirs(SPLIT_DIR, exist_ok=True)
    
    # Path to save the split JSON file
    split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
    split_data = {'train': train_files, 'test': test_files}
    
    # Save the split data to a JSON file
    with open(split_file, 'w') as f:
        json.dump(split_data, f)

    print(f"Training & Testing data successfully split into {split_file}")

    return train_files, test_files

def register_datasets(dataset_info, dataset_name, test_size=0.2):
    """
    Registers the selected dataset in the Detectron2 framework.

    Parameters:
    - dataset_info: Dictionary containing dataset names and their info.
    - dataset_name: Name of the dataset to register.
    - test_size: Proportion of the dataset to include in the test split.
    """
    if dataset_name not in dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_info.")

    img_dir, label_dir, thing_classes = dataset_info[dataset_name]
    
    print(f"Processing dataset: {dataset_name}, Info: {dataset_info[dataset_name]}")

    # Load or split the dataset
    split_file = os.path.join(SPLIT_DIR, f"{dataset_name}_split.json")
    category_key = dataset_name
    
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        train_files = split_data['train']
        test_files = split_data['test']
    else:
        # Create split data if it doesn't exist
        train_files, test_files = split_dataset(img_dir, dataset_name, test_size=test_size)
        split_data = {'train': train_files, 'test': test_files}
        os.makedirs(SPLIT_DIR, exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(split_data, f)
        print(f"Split created and saved at {split_file}")

    # Register training dataset
    DatasetCatalog.register(
        f"{dataset_name}_train",
        lambda img_dir=img_dir, label_dir=label_dir, files=train_files:
        get_split_dicts(img_dir, label_dir, files, CATEGORY_JSON, category_key)
    )
    MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

    # Register testing dataset
    DatasetCatalog.register(
        f"{dataset_name}_test",
        lambda img_dir=img_dir, label_dir=label_dir, files=test_files:
        get_split_dicts(img_dir, label_dir, files, CATEGORY_JSON, category_key)
    )
    MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)

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
    
    if category_key not in dataset_info:
        raise ValueError(f"Category key '{category_key}' not found in JSON")
    
    category_names = dataset_info[category_key][2]  # Extract category names from the JSON
    category_name_to_id = {name: idx for idx, name in enumerate(category_names)}
    
    print(f"Category Mapping: {category_name_to_id}")

    dataset_dicts = []
    for idx, file in enumerate(files):
        json_file = os.path.join(label_dir, file)
        with open(json_file) as f:
            imgs_anns = json.load(f)

        record = {}
        filename = os.path.join(img_dir, imgs_anns["metadata"]["name"])
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = imgs_anns["metadata"]["height"]
        record["width"] = imgs_anns["metadata"]["width"]
        
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
                print(f"Warning: Category Name Not Found: {categoryName}")
                continue

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

def get_trained_model_paths(base_dir):
    """
    Retrieves paths to trained models in a given base directory.

    Parameters:
    - base_dir: Directory containing trained models.

    Returns:
    - model_paths: Dictionary with dataset names as keys and model paths as values.
    """
    model_paths = {}
    for dataset_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, dataset_name)
        model_path = os.path.join(model_dir, "model_final.pth")
        if os.path.exists(model_path):
            model_paths[dataset_name] = model_path
    return model_paths

def load_model(cfg, model_path, dataset_name):
    """
    Loads a trained model with a specific configuration.

    Parameters:
    - cfg: Configuration object for the model.
    - model_path: Path to the trained model.
    - dataset_name: Name of the dataset for metadata.

    Returns:
    - predictor: Loaded predictor object.
    """
    cfg.MODEL.WEIGHTS = model_path
    thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    predictor = DefaultPredictor(cfg)
    return predictor

def choose_and_use_model(model_paths, dataset_name, threshold):
    """
    Selects and loads a trained model for a specific dataset.

    Parameters:
    - model_paths: Dictionary of model paths.
    - dataset_name: Name of the dataset for which the model is used.
    - threshold: Detection threshold for ROI heads score.

    Returns:
    - predictor: Predictor object for inference.
    """
    if dataset_name not in model_paths:
        print(f"No model found for dataset {dataset_name}")
        return None

    model_path = model_paths[dataset_name]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    predictor = load_model(cfg, model_path, dataset_name)
    return predictor

def read_dataset_info(file_path):
    """
    Reads dataset information from a JSON file.

    Parameters:
    - file_path: Path to the JSON file.

    Returns:
    - dataset_info: Dictionary with dataset information.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        # Convert list values back to tuples for consistency with the original data
        dataset_info = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        print("Dataset Info:", dataset_info)
    return dataset_info
