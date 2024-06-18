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
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import csv

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

def choose_and_use_model(model_paths, dataset_name):
    """
    Selects and loads a trained model for a specific dataset.

    Parameters:
    - model_paths: Dictionary of model paths.
    - dataset_name: Name of the dataset for which the model is used.

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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45

    predictor = load_model(cfg, model_path, dataset_name)
    return predictor

def read_dataset_info(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        # Convert list values back to tuples for consistency with the original data
        dataset_info = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    return dataset_info

def get_split_dicts(img_dir, label_dir, files):
    """
    Generates a list of dictionaries for Detectron2 dataset registration.
    
    Parameters:
    - img_dir: Directory containing images.
    - label_dir: Directory containing labels.
    - files: List of label files to process.
    
    Returns:
    - dataset_dicts: List of dictionaries with image and annotation data.
    """
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

            if "throat" in categoryName:
                category_id = 0
            elif "pore" in categoryName:
                category_id = 1
            else:
                raise ValueError("Category Name Not Found: " + categoryName)

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
            get_split_dicts(img_dir, label_dir, files)
        )
        MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

        # Register testing dataset
        DatasetCatalog.register(
            f"{dataset_name}_test",
            lambda img_dir=img_dir, label_dir=label_dir, files=test_files:
            get_split_dicts(img_dir, label_dir, files)
        )
        MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)

        print(f"Checking name: {dataset_name}, img: {img_dir}, label: {label_dir}, classes: {thing_classes}")

def evaluate_model(dataset_name, output_dir, visualize=False):
    """
    Evaluates a trained model on the specified dataset.

    Parameters:
    - dataset_name: Name of the dataset to evaluate.
    - output_dir: Directory to save evaluation results.
    - visualize: Boolean, if True, save visualizations of predictions.

    Returns:
    - metrics: Dictionary containing evaluation metrics.
    """
    dataset_info = read_dataset_info('/home/deamoon_uw_nn/uw-com-vision/dataset_info.json')
    register_datasets(dataset_info)
    
    trained_model_paths = get_trained_model_paths("/home/deamoon_uw_nn/split_dir")
    selected_model_dataset = dataset_name  # User-selected model
    predictor = choose_and_use_model(trained_model_paths, selected_model_dataset)
    
    # model_path = os.path.join("./trained_models", dataset_name, "model_final.pth")
    

    # Load model configuration
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_path
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    # cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # predictor = DefaultPredictor(cfg)
    
    # Prepare the evaluation data loader
    evaluator = COCOEvaluator(f"{dataset_name}_test", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, f"{dataset_name}_test")
    
    # Run evaluation
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Print metrics
    print(f"Evaluation metrics: {metrics}")

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, "metrics.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['metric', 'value']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({'metric': key, 'value': value})
    
    print(f"Metrics saved to {csv_path}")

    return metrics

def visualize_predictions(predictor, dataset_name, output_dir):
    """
    Visualizes predictions on the evaluation dataset.

    Parameters:
    - predictor: The trained predictor object.
    - dataset_name: Name of the dataset.
    - output_dir: Directory to save visualizations.
    """
    dataset_dicts = DatasetCatalog.get(f"{dataset_name}_test")
    metadata = MetadataCatalog.get(f"{dataset_name}_test")

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_output = v.get_image()[:, :, ::-1]

        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, os.path.basename(d["file_name"]))
        cv2.imwrite(vis_path, vis_output)
        print(f"Saved visualization to {vis_path}")

def calculate_metrics(true_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculates precision, recall, F1 score, and IoU.

    Parameters:
    - true_boxes: Ground truth bounding boxes.
    - pred_boxes: Predicted bounding boxes.
    - iou_threshold: IoU threshold to consider a prediction as true positive.

    Returns:
    - metrics: Dictionary containing calculated metrics.
    """
    # Flatten the lists
    all_true_boxes = [box for sublist in true_boxes for box in sublist]
    all_pred_boxes = [box for sublist in pred_boxes for box in sublist]

    # Calculate IoU for each pair of true and predicted boxes
    ious = []
    for true_box in all_true_boxes:
        for pred_box in all_pred_boxes:
            iou = compute_iou(true_box, pred_box)
            if iou >= iou_threshold:
                ious.append(iou)

    # Calculate metrics based on IoU
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_boxes, all_pred_boxes, average='binary', pos_label=1)
    accuracy = accuracy_score(all_true_boxes, all_pred_boxes)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "ious": ious
    }
    return metrics

def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - boxA: First bounding box.
    - boxB: Second bounding box.

    Returns:
    - iou: Intersection over Union (IoU) value.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Example usage
# if __name__ == "__main__":
#     dataset_name = "polyhipes"  # Example dataset name
#     model_path = "./trained_models/polyhipes/model_final.pth"  # Example path to model
#     output_dir = "./evaluation_results"  # Directory to save evaluation results and visualizations

#     # Evaluate the model
#     metrics = evaluate_model(dataset_name, model_path, output_dir, visualize=True)
#     print(metrics)
