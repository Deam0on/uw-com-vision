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
