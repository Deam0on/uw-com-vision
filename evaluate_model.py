import os
import json
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_model(dataset_name, model_path, output_dir, visualize=False):
    """
    Evaluates a trained model on the specified dataset.

    Parameters:
    - dataset_name: Name of the dataset to evaluate.
    - model_path: Path to the trained model weights.
    - output_dir: Directory to save evaluation results.
    - visualize: Boolean, if True, save visualizations of predictions.

    Returns:
    - metrics: Dictionary containing evaluation metrics.
    """
    # Load model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.MODEL.DEVICE = "cuda"  # Use GPU for inference if available

    predictor = DefaultPredictor(cfg)
    
    # Prepare the evaluation data loader
    evaluator = COCOEvaluator(f"{dataset_name}_test", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, f"{dataset_name}_test")
    
    # Run evaluation
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Print metrics
    print(f"Evaluation metrics: {metrics}")

    if visualize:
        visualize_predictions(predictor, dataset_name, output_dir)

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
if __name__ == "__main__":
    dataset_name = "polyhipes"  # Example dataset name
    model_path = "./trained_models/polyhipes/model_final.pth"  # Example path to model
    output_dir = "./evaluation_results"  # Directory to save evaluation results and visualizations

    # Evaluate the model
    metrics = evaluate_model(dataset_name, model_path, output_dir, visualize=True)
    print(metrics)
