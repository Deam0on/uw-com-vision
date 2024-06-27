import os
import json
import numpy as np
import torch
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import csv
from data_preparation import split_dataset, register_datasets, get_split_dicts
from data_preparation import get_trained_model_paths, load_model, choose_and_use_model, read_dataset_info

# def choose_and_use_model(model_paths, dataset_name, threshold):
#     """
#     Selects and loads a trained model for a specific dataset.

#     Parameters:
#     - model_paths: Dictionary of model paths.
#     - dataset_name: Name of the dataset for which the model is used.
#     - threshold: Detection threshold for ROI heads score.

#     Returns:
#     - predictor: Predictor object for inference.
#     """
#     if dataset_name not in model_paths:
#         print(f"No model found for dataset {dataset_name}")
#         return None

#     model_path = model_paths[dataset_name]
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
#     cfg.MODEL.DEVICE = "cuda"
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Set threshold here

#     predictor = load_model(cfg, model_path, dataset_name)
#     return predictor

# def load_model(cfg, model_path, dataset_name):
#     """
#     Loads a trained model with a specific configuration.

#     Parameters:
#     - cfg: Configuration object for the model.
#     - model_path: Path to the trained model.
#     - dataset_name: Name of the dataset for metadata.

#     Returns:
#     - predictor: Loaded predictor object.
#     """
#     cfg.MODEL.WEIGHTS = model_path
#     thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
#     predictor = DefaultPredictor(cfg)
#     return predictor

# def read_dataset_info(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#         dataset_info = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
#     return dataset_info

# def get_split_dicts(img_dir, label_dir, files, category_json, category_key):
#     dataset_info = read_dataset_info(category_json)
    
#     if category_key not in dataset_info:
#         raise ValueError(f"Category key '{category_key}' not found in JSON")
    
#     category_names = dataset_info[category_key][2]
#     category_name_to_id = {name: idx for idx, name in enumerate(category_names)}

#     dataset_dicts = []
#     idx = 0
#     for file in files:
#         json_file = os.path.join(label_dir, file)
#         if not os.path.exists(json_file):
#             print(f"Label file not found: {json_file}")
#             continue
        
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

#             if categoryName in category_name_to_id:
#                 category_id = category_name_to_id[categoryName]
#             else:
#                 raise ValueError(f"Category Name Not Found: {categoryName}")

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

# def get_trained_model_paths(base_dir):
#     """
#     Retrieves paths to trained models in a given base directory.

#     Parameters:
#     - base_dir: Directory containing trained models.

#     Returns:
#     - model_paths: Dictionary with dataset names as keys and model paths as values.
#     """
#     model_paths = {}
#     for dataset_name in os.listdir(base_dir):
#         model_dir = os.path.join(base_dir, dataset_name)
#         model_path = os.path.join(model_dir, "model_final.pth")
#         if os.path.exists(model_path):
#             model_paths[dataset_name] = model_path
#     return model_paths

# def register_datasets(dataset_name, dataset_info, test_size=0.2):
#     img_dir, label_dir, thing_classes = dataset_info[dataset_name]
    
#     split_dir = "/home/deamoon_uw_nn/split_dir/"
#     split_file = os.path.join(split_dir, f"{dataset_name}_split.json")
#     category_json = "/home/deamoon_uw_nn/uw-com-vision/dataset_info.json"
#     category_key = dataset_name
    
#     if os.path.exists(split_file):
#         with open(split_file, 'r') as f:
#             split_data = json.load(f)
#         train_files = split_data['train']
#         test_files = split_data['test']
#     else:
#         print(f"No split found at {split_file}")

#     DatasetCatalog.register(
#         f"{dataset_name}_train",
#         lambda img_dir=img_dir, label_dir=label_dir, files=train_files:
#         get_split_dicts(img_dir, label_dir, files, category_json, category_key)
#     )
#     MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

#     DatasetCatalog.register(
#         f"{dataset_name}_test",
#         lambda img_dir=img_dir, label_dir=label_dir, files=test_files:
#         get_split_dicts(img_dir, label_dir, files, category_json, category_key)
#     )
#     MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)

def evaluate_model(dataset_name, output_dir, visualize=False):
    dataset_info = read_dataset_info('/home/deamoon_uw_nn/uw-com-vision/dataset_info.json')
    register_datasets(dataset_name, dataset_info)
    
    trained_model_paths = get_trained_model_paths("/home/deamoon_uw_nn/split_dir")

    threshold = 0.45
    
    predictor = choose_and_use_model(trained_model_paths, dataset_name, threshold)
    
    cfg = get_cfg()

    evaluator = COCOEvaluator(f"{dataset_name}_test", cfg, False, output_dir=output_dir)
    
    # Ensure no cached data is used
    coco_format_cache = os.path.join("/home/deamoon_uw_nn/split_dir", f"{dataset_name}_test_coco_format.json")
    if os.path.exists(coco_format_cache):
        os.remove(coco_format_cache)

    val_loader = build_detection_test_loader(cfg, f"{dataset_name}_test")

    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(f"Evaluation metrics: {metrics}")

    csv_path = os.path.join(output_dir, "metrics.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['metric', 'value']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({'metric': key, 'value': value})
    
    print(f"Metrics saved to {csv_path}")

    if visualize:
        visualize_predictions(predictor, dataset_name, output_dir)

def visualize_predictions(predictor, dataset_name, output_dir):
    dataset_dicts = DatasetCatalog.get(f"{dataset_name}_test")
    metadata = MetadataCatalog.get(f"{dataset_name}_test")

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_output = v.get_image()[:, :, ::-1]

        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, os.path.basename(d["file_name"]))
        cv2.imwrite(vis_path, vis_output)
        print(f"Saved visualization to {vis_path}")

# The functions should now work as expected when called from `main.py`.
