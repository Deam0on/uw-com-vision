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
import scipy
from scipy.spatial import distance as dist
import imutils
from imutils import perspective, contours
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
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import pycocotools.mask as mask_util
from skimage.measure import find_contours, label
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, erosion
from google.cloud import storage
import easyocr
import re
from numpy import sqrt
from data_preparation import split_dataset

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

# def choose_and_use_model(model_paths, dataset_name):
#     """
#     Selects and loads a trained model for a specific dataset.

#     Parameters:
#     - model_paths: Dictionary of model paths.
#     - dataset_name: Name of the dataset for which the model is used.

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
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65

#     predictor = load_model(cfg, model_path, dataset_name)
#     return predictor

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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # Set threshold here

    predictor = load_model(cfg, model_path, dataset_name)
    return predictor

def get_image_folder_path(base_path='/home/deamoon_uw_nn/DATASET/INFERENCE/'):
    """
    Determines the path to the folder containing images for inference.

    Parameters:
    - base_path: Base path where the INFERENCE folder is located.

    Returns:
    - str: Path to the folder containing the images.
    """
    # Define the two possible paths
    inference_path = os.path.join(base_path)
    upload_path = os.path.join(base_path, 'UPLOAD')

    # Check if the INFERENCE folder contains images
    if any(os.path.isfile(os.path.join(inference_path, f)) for f in os.listdir(inference_path)):
        return inference_path

    # Check if the UPLOAD subfolder contains images
    elif os.path.exists(upload_path) and any(os.path.isfile(os.path.join(upload_path, f)) for f in os.listdir(upload_path)):
        return upload_path

    # If no images found in either folder, raise an exception
    else:
        raise FileNotFoundError("No images found in INFERENCE or INFERENCE/UPLOAD folders.")

def binary_mask_to_rle(binary_mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE).

    Parameters:
    - binary_mask: 2D binary mask.

    Returns:
    - rle: Dictionary with RLE counts and mask size.
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def rle_encode(img):
    """
    Encodes a binary image into Run-Length Encoding (RLE).

    Parameters:
    - img: Numpy array, 1 - mask, 0 - background.

    Returns:
    - str: Run length as string formatted.
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_masks(fn, predictor):
    """
    Gets predicted masks for an image using a trained model.

    Parameters:
    - fn: File name of the image.
    - predictor: Predictor object for inference.

    Returns:
    - res: List of RLE encoded masks.
    """
    im = cv2.imread(fn)
    pred = predictor(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    for mask in pred_masks:
        mask = mask * (1 - used)
        if mask.sum() >= MIN_PIXELS[pred_class]:
            used += mask
            res.append(rle_encode(mask))
    return res

def rle_decode(mask_rle, shape):
    """
    Decodes Run-Length Encoding (RLE) into a binary mask.

    Parameters:
    - mask_rle: Run-length as string formatted (start length).
    - shape: (height, width) of array to return.

    Returns:
    - img: Numpy array, 1 - mask, 0 - background.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def rle_encoding(x):
    """
    Encodes a binary array into run-length encoding.
    
    Parameters:
    - x: Numpy array, 1 - mask, 0 - background.

    Returns:
    - list: Run-length encoding list.
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def postprocess_masks(ori_mask, ori_score, image, min_crys_size=2):
    """
    Post-processes masks by removing overlaps, filling small holes, and smoothing boundaries.

    Parameters:
    - ori_mask: Original mask predictions.
    - ori_score: Confidence scores for the masks.
    - image: Original image for reference.
    - min_crys_size: Minimum size for valid masks.

    Returns:
    - masks: List of processed masks.
    """
    image = image[:, :, ::-1]
    height, width = image.shape[:2]

    score_threshold = 0.5

    if len(ori_mask) == 0 or ori_score.all() < score_threshold:
        return []

    keep_ind = np.where(np.sum(ori_mask, axis=(0, 1)) > min_crys_size)[0]
    if len(keep_ind) < len(ori_mask):
        if keep_ind.shape[0] != 0:
            ori_mask = ori_mask[:keep_ind.shape[0]]
            ori_score = ori_score[:keep_ind.shape[0]]
        else:
            return []

    overlap = np.zeros([height, width])
    masks = []

    # Removes overlaps from masks with lower scores
    for i in range(len(ori_mask)):
        mask = binary_fill_holes(ori_mask[i]).astype(np.uint8)
        mask = erosion(dilation(mask))
        overlap += mask
        mask[overlap > 1] = 0
        out_label = label(mask)
        if out_label.max() > 1:
            mask[:] = 0
        masks.append(mask)

    return masks

def midpoint(ptA, ptB):
    """
    Computes the midpoint between two points.

    Parameters:
    - ptA: Tuple representing the first point.
    - ptB: Tuple representing the second point.

    Returns:
    - tuple: Midpoint coordinates.
    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def GetInference(predictor, im, x_pred, metadata, test_img):
    """
    Performs inference on an image and saves the predicted instances.

    Assumes the global variables `im`, `predictor`, `metadata`, `test_img`, and `x_pred` are defined.
    """
    outputs = predictor(im)

    # Get all instances
    inst_out = outputs['instances']

    # Filter instances by predicted class
    filtered_instances = inst_out[inst_out.pred_classes == x_pred]
    
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=1,
                   instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(filtered_instances.to("cpu"))
    cv2.imwrite(test_img + '_' + str(x_pred) + "__pred.png", out.get_image()[:, :, ::-1])

def GetCounts(predictor, im, TList, PList):
    """
    Counts the number of instances for each class in the image.

    Assumes the global variables `im`, `predictor`, `TList`, and `PList` are defined.
    """
    outputs = predictor(im)
    classes = outputs["instances"].pred_classes.to("cpu").numpy()
    TotalCount = sum(classes == 0) + sum(classes == 1)
    TCount = sum(classes == 0)
    PCount = sum(classes == 1)
    TList.append(TCount)
    PList.append(PCount)

def read_dataset_info(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        # Convert list values back to tuples for consistency with the original data
        dataset_info = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    return dataset_info

def rgb_to_hsv(r, g, b):
    MAX_PIXEL_VALUE = 255.0

    r = r / MAX_PIXEL_VALUE
    g = g / MAX_PIXEL_VALUE
    b = b / MAX_PIXEL_VALUE

    max_val = max(r, g, b)
    min_val = min(r, g, b)
    v = max_val

    if max_val == 0.0:
        s = 0
        h = 0
    elif (max_val - min_val) == 0.0:
        s = 0
        h = 0
    else:
        s = (max_val - min_val) / max_val

        if max_val == r:
            h = 60 * ((g - b) / (max_val - min_val)) + 0
        elif max_val == g:
            h = 60 * ((b - r) / (max_val - min_val)) + 120
        else:
            h = 60 * ((r - g) / (max_val - min_val)) + 240

    if h < 0:
        h = h + 360.0

    h = h / 2
    s = s * MAX_PIXEL_VALUE
    v = v * MAX_PIXEL_VALUE

    return h, s, v

# def rgb_to_wavelength(b, g, r):
#     hsv_image = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
#     h, s, v = hsv_image
#     wavelength = hue_to_wavelength(h)
#     return wavelength

def hue_to_wavelength(hue):
    # There is nothing corresponding to magenta in the light spectrum,
    # So let's assume that we only use hue values between 0 and 270.
    assert hue >= 0
    assert hue <= 270

    # Estimating that the usable part of the visible spectrum is 450-620nm,
    # with wavelength (in nm) and hue value (in degrees), you can improvise this:
    wavelength = 620 - 170 / 270 * hue
    return wavelength

def rgb_to_wavelength(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    wavelength = hue_to_wavelength(h)
    return wavelength

def detect_arrows(image):
    """
    Detect arrows in the image and compute their directions.
    
    Parameters:
    - image: Input image.
    
    Returns:
    - flow_vectors: List of detected flow vectors (directions).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    flow_vectors = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Filter out small contours
            continue
        
        # Approximate the contour
        # epsilon = 0.03 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # if len(approx) == 7:  # Arrows typically have 7 points
        #     # Fit a line to the contour points
        #     [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
        #     # Calculate the direction vector
        #     direction = (vx, vy)
        #     flow_vectors.append(direction)
    # Fit a line to the contour points
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate the direction vector
        direction = (vx[0], vy[0])
        flow_vectors.append(direction)
    
    return flow_vectors


# def run_inference(dataset_name, output_dir, visualize=False, threshold=0.65):
#     """
#     Runs inference on images in the specified directory using the provided model.

#     Parameters:
#     - dataset_name: Name of the dataset.
#     - output_dir: Directory to save inference results.
#     - visualize: Boolean, if True, save visualizations of predictions.
#     """
#     dataset_info = read_dataset_info('/home/deamoon_uw_nn/uw-com-vision/dataset_info.json')
#     register_datasets(dataset_info)
    
#     trained_model_paths = get_trained_model_paths("/home/deamoon_uw_nn/split_dir")
#     selected_model_dataset = dataset_name  # User-selected model
#     # predictor = choose_and_use_model(trained_model_paths, selected_model_dataset)
#     predictor = choose_and_use_model(trained_model_paths, selected_model_dataset, threshold)
    
    
#     metadata = MetadataCatalog.get(f"{dataset_name}_train")
    
#     image_folder_path = get_image_folder_path()
    
#     # Path to save outputs
#     path = output_dir
#     os.makedirs(path, exist_ok=True)
#     inpath = image_folder_path
#     images_name = [f for f in os.listdir(inpath) if f.endswith('.tif')]
    
#     Img_ID = []
#     EncodedPixels = []

#     test_count = 0
        
#     conv = lambda l: ' '.join(map(str, l))
    
#     for name in images_name:
#         image = cv2.imread(os.path.join(inpath, name))
#         outputs = predictor(image)
#         masks = postprocess_masks(
#             np.asarray(outputs["instances"].to('cpu')._fields['pred_masks']),
#             outputs["instances"].to('cpu')._fields['scores'].numpy(), image)
    
#         if masks:  # If any objects are detected in this image
#             for i in range(len(masks)):  # Loop all instances
#                 Img_ID.append(name.replace('.tif', ''))
#                 EncodedPixels.append(conv(rle_encoding(masks[i])))
    
#     # Save inference results
#     df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
#     df.to_csv(os.path.join(path, "R50_flip_results.csv"), index=False, sep=',')

#     for x_pred in [0, 1]:
#         TList = []
#         PList = []
#         csv_filename = f'results_x_pred_{x_pred}.csv'
#         test_img_path = image_folder_path
    
#         # Open CSV file before processing images
#         with open(csv_filename, 'w', newline='') as csvfile:
#             csvwriter = csv.writer(csvfile)

#             if dataset_name != 'hw_patterns':
#                 csvwriter.writerow(['length', 'width', 'circularED', 'aspectRatio', 'circularity', 'chords', 'ferret', 'round', 'sphere', 'psum', 'name'])
#             else:
#                 csvwriter.writerow(['length', 'width', 'E_major', 'E_minor', 'Eccentricity', 'min_velocity', 'avg_velocity', 'max_velocity', 'name'])

    
#             for test_img in os.listdir(test_img_path):
#                 input_path = os.path.join(test_img_path, test_img)
#                 im = cv2.imread(input_path)
    
#                 # Convert image to grayscale
#                 gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
#                 # Use canny edge detection
#                 edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#                 if dataset_name != 'hw_patterns':
#                     # Execute this block for datasets other than 'hw_patterns'
#                     reader = easyocr.Reader(['en'])
#                     result = reader.readtext(gray, detail=0, paragraph=False, contrast_ths=0.85, adjust_contrast=0.85, add_margin=0.25, width_ths=0.25, decoder='beamsearch')
#                     if result:  # Ensure result is not empty
#                         pxum_r = result[0]
#                         psum = re.sub("[^0-9]", "", pxum_r)
#                     else:
#                         pxum_r = ''
#                         psum = '0'

#                     lines_list = []
#                     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=1)
        
#                     if lines is not None:
#                         for points in lines:
#                             x1, y1, x2, y2 = points[0]
#                             cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                             lines_list.append([(x1, y1), (x2, y2)])
#                             scale_len = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#                             um_pix = float(psum) / scale_len
#                     else:
#                         um_pix = 1
#                         psum = '0'
#                 else:
#                     # Placeholder: You can add any specific handling for 'hw_patterns' here if needed.
#                     um_pix = 1
#                     psum = '0'
    
#                 GetInference(predictor, im, x_pred, metadata, test_img)  # Ensure this function is correctly defined elsewhere
#                 GetCounts(predictor, im, TList, PList)  # Ensure this function is correctly defined elsewhere
    
#                 outputs = predictor(im)
#                 inst_out = outputs['instances']
#                 filtered_instances = inst_out[inst_out.pred_classes == x_pred]
#                 mask_array = filtered_instances.pred_masks.to("cpu").numpy()
#                 num_instances = mask_array.shape[0]
#                 mask_array = np.moveaxis(mask_array, 0, -1)
#                 output = np.zeros_like(im)

#                 hsv_image_global = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#                 global_velocities = hsv_image_global[..., 2]  # Normalize the V channel to [0, 1]
                
#                 global_min_velocity = np.min(global_velocities)
#                 global_max_velocity = np.max(global_velocities)

#                 for i in range(num_instances):
#                     # Initialize a new output array for each mask
#                     single_output = np.zeros_like(output)
#                     mask = mask_array[:, :, i:(i + 1)]
#                     single_output = np.where(mask == True, 255, single_output)
                    
#                     # Save each mask image separately for debugging
#                     mask_filename = os.path.join(output_dir, f'mask_{i}.jpg')
#                     cv2.imwrite(mask_filename, single_output)
                    
#                     # Convert the single mask to grayscale for contour detection
#                     single_im_mask = cv2.cvtColor(single_output, cv2.COLOR_BGR2GRAY)
#                     single_cnts = cv2.findContours(single_im_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     single_cnts = imutils.grab_contours(single_cnts)
                
#                     for c in single_cnts:
#                         pixelsPerMetric = 1  # or 0.85, correction
#                         if cv2.contourArea(c) < 100:
#                             continue
#                         area = cv2.contourArea(c)
#                         perimeter = cv2.arcLength(c, True)
                
#                         orig = single_im_mask.copy()
#                         box = cv2.minAreaRect(c)
#                         box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
#                         box = np.array(box, dtype="int")
#                         box = perspective.order_points(box)
#                         cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
#                         for (x, y) in box:
#                             cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
#                         (tl, tr, br, bl) = box
#                         (tltrX, tltrY) = midpoint(tl, tr)
#                         (blbrX, blbrY) = midpoint(bl, br)
#                         (tlblX, tlblY) = midpoint(tl, bl)
#                         (trbrX, trbrY) = midpoint(tr, br)
#                         dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
#                         dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
#                         dimA = dA / pixelsPerMetric
#                         dimB = dB / pixelsPerMetric

#                         dimArea = area / pixelsPerMetric
#                         dimPerimeter = perimeter / pixelsPerMetric
#                         diaFeret = max(dimA, dimB)
#                         # Execute this block for datasets other than 'hw_patterns'
#                         if (dimA and dimB) != 0:
#                             Aspect_Ratio = max(dimB, dimA) / min(dimA, dimB)
#                         else:
#                             Aspect_Ratio = 0
#                         Length = min(dimA, dimB) * um_pix
#                         Width = max(dimA, dimB) * um_pix

#                         ellipse = cv2.fitEllipse(c)
#                         (x, y), (major_axis, minor_axis), angle = ellipse
                        
#                         if major_axis > minor_axis:
#                             a = major_axis / 2.0
#                             b = minor_axis / 2.0
#                         else:
#                             a = minor_axis / 2.0
#                             b = major_axis / 2.0
#                         eccentricity = np.sqrt(1 - (b**2 / a**2))
    
#                         # Assuming pixelsPerMetric and um_pix are defined earlier in the code
#                         major_axis_length = major_axis / pixelsPerMetric * um_pix
#                         minor_axis_length = minor_axis / pixelsPerMetric * um_pix

#                         if dataset_name != 'hw_patterns':
#                             CircularED = np.sqrt(4 * area / np.pi) * um_pix
#                             Chords = cv2.arcLength(c, True) * um_pix
#                             Roundness = 1 / Aspect_Ratio if Aspect_Ratio != 0 else 0
#                             Sphericity = (2 * np.sqrt(np.pi * dimArea)) / dimPerimeter * um_pix
#                             Circularity = 4 * np.pi * (dimArea / (dimPerimeter) ** 2) * um_pix
#                             Feret_diam = diaFeret * um_pix

#                             csvwriter.writerow([Length, Width, CircularED, Aspect_Ratio, Circularity, Chords, Feret_diam, Roundness, Sphericity, psum, test_img])
#                         else:
#                             hsv_image_global = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#                             global_velocities = hsv_image_global[..., 2] / 255.0  # Normalize the V channel to [0, 1]
                            
#                             global_min_velocity = np.min(global_velocities)
#                             global_max_velocity = np.max(global_velocities)
                            
#                             # Normalize the global velocities
#                             normalized_global_velocities = (global_velocities - global_min_velocity) / (global_max_velocity - global_min_velocity)
                            
#                             # Now process each contour (mask) as before
#                             mask = np.zeros(im.shape[:2], dtype=np.uint8)
#                             cv2.drawContours(mask, [c], -1, 255, -1)
#                             masked_image = cv2.bitwise_and(im, im, mask=mask)
                            
#                             # Convert the masked image to HSV
#                             hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
                            
#                             # Extract the V channel for velocities
#                             velocities = hsv_image[..., 2] / 255.0  # Normalize the V channel to [0, 1]
#                             velocities = velocities[mask == 255]  # Only consider the velocities within the mask
                            
#                             # Normalize velocities within the mask based on the global min and max
#                             normalized_velocities = (velocities - global_min_velocity) / (global_max_velocity - global_min_velocity)
                            
#                             # Compute the min, average, and max velocities within the mask
#                             min_velocity = np.min(normalized_velocities)
#                             avg_velocity = np.mean(normalized_velocities)
#                             max_velocity = np.max(normalized_velocities)

#                             csvwriter.writerow([Length, Width, major_axis_length, minor_axis_length, eccentricity, min_velocity, avg_velocity, max_velocity, test_img])

def run_inference(dataset_name, output_dir, visualize=False, threshold=0.65):
    """
    Runs inference on images in the specified directory using the provided model.

    Parameters:
    - dataset_name: Name of the dataset.
    - output_dir: Directory to save inference results.
    - visualize: Boolean, if True, save visualizations of predictions.
    """
    dataset_info = read_dataset_info('/home/deamoon_uw_nn/uw-com-vision/dataset_info.json')
    register_datasets(dataset_info)
    
    trained_model_paths = get_trained_model_paths("/home/deamoon_uw_nn/split_dir")
    selected_model_dataset = dataset_name  # User-selected model
    predictor = choose_and_use_model(trained_model_paths, selected_model_dataset, threshold)
    
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    
    image_folder_path = get_image_folder_path()
    
    # Path to save outputs
    path = output_dir
    os.makedirs(path, exist_ok=True)
    inpath = image_folder_path
    images_name = [f for f in os.listdir(inpath) if f.endswith('.tif')]
    
    Img_ID = []
    EncodedPixels = []

    conv = lambda l: ' '.join(map(str, l))

    for name in images_name:
        image = cv2.imread(os.path.join(inpath, name))
        outputs = predictor(image)
        masks = postprocess_masks(
            np.asarray(outputs["instances"].to('cpu')._fields['pred_masks']),
            outputs["instances"].to('cpu')._fields['scores'].numpy(), image)
    
        if masks:
            for i in range(len(masks)):
                Img_ID.append(name.replace('.tif', ''))
                EncodedPixels.append(conv(rle_encoding(masks[i])))
    
    df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
    df.to_csv(os.path.join(path, "R50_flip_results.csv"), index=False, sep=',')

    for x_pred in [0, 1]:
        TList = []
        PList = []
        csv_filename = f'results_x_pred_{x_pred}.csv'
        test_img_path = image_folder_path
    
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            if dataset_name != 'hw_patterns':
                csvwriter.writerow(['length', 'width', 'circularED', 'aspectRatio', 'circularity', 'chords', 'ferret', 'round', 'sphere', 'psum', 'name'])
            else:
                csvwriter.writerow(['length', 'width', 'E_major', 'E_minor', 'Eccentricity', 'Global_min_velocity', 'avg_velocity', 'Global_max_velocity', 'name'])

    
            for test_img in os.listdir(test_img_path):
                input_path = os.path.join(test_img_path, test_img)
                im = cv2.imread(input_path)
    
                # Convert image to grayscale
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
                # Use canny edge detection
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)

                if dataset_name != 'hw_patterns':
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext(gray, detail=0, paragraph=False, contrast_ths=0.85, adjust_contrast=0.85, add_margin=0.25, width_ths=0.25, decoder='beamsearch')
                    if result:
                        pxum_r = result[0]
                        psum = re.sub("[^0-9]", "", pxum_r)
                    else:
                        pxum_r = ''
                        psum = '0'

                    lines_list = []
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=1)
        
                    if lines is not None:
                        for points in lines:
                            x1, y1, x2, y2 = points[0]
                            cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            lines_list.append([(x1, y1), (x2, y2)])
                            scale_len = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            um_pix = float(psum) / scale_len
                    else:
                        um_pix = 1
                        psum = '0'
                else:
                    um_pix = 1
                    psum = '0'
    
                GetInference(predictor, im, x_pred, metadata, test_img)
                GetCounts(predictor, im, TList, PList)
    
                outputs = predictor(im)
                inst_out = outputs['instances']
                filtered_instances = inst_out[inst_out.pred_classes == x_pred]
                mask_array = filtered_instances.pred_masks.to("cpu").numpy()
                num_instances = mask_array.shape[0]
                mask_array = np.moveaxis(mask_array, 0, -1)
                output = np.zeros_like(im)

                global_min_wavelength = float('inf')
                global_max_wavelength = float('-inf')

                for i in range(im.shape[0]):
                    for j in range(im.shape[1]):
                        b, g, r = im[i, j]
                        wavelength = rgb_to_wavelength(b, g, r)
                        if wavelength < global_min_wavelength:
                            global_min_wavelength = wavelength
                        if wavelength > global_max_wavelength:
                            global_max_wavelength = wavelength

                # print(f"Global min wavelength: {global_min_wavelength}, max wavelength: {global_max_wavelength}")

                for i in range(num_instances):
                    single_output = np.zeros_like(output)
                    mask = mask_array[:, :, i:(i + 1)]
                    single_output = np.where(mask == True, 255, single_output)
                    
                    mask_filename = os.path.join(output_dir, f'mask_{i}.jpg')
                    cv2.imwrite(mask_filename, single_output)
                    
                    single_im_mask = cv2.cvtColor(single_output, cv2.COLOR_BGR2GRAY)
                    single_cnts = cv2.findContours(single_im_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    single_cnts = imutils.grab_contours(single_cnts)
                
                    for c in single_cnts:
                        pixelsPerMetric = 1
                        if cv2.contourArea(c) < 100:
                            continue
                        area = cv2.contourArea(c)
                        perimeter = cv2.arcLength(c, True)
                
                        orig = single_im_mask.copy()
                        box = cv2.minAreaRect(c)
                        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                        box = np.array(box, dtype="int")
                        box = perspective.order_points(box)
                        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
                        for (x, y) in box:
                            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                        (tl, tr, br, bl) = box
                        (tltrX, tltrY) = midpoint(tl, tr)
                        (blbrX, blbrY) = midpoint(bl, br)
                        (tlblX, tlblY) = midpoint(tl, bl)
                        (trbrX, trbrY) = midpoint(tr, br)
                        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                        dimA = dA / pixelsPerMetric
                        dimB = dB / pixelsPerMetric

                        dimArea = area / pixelsPerMetric
                        dimPerimeter = perimeter / pixelsPerMetric
                        diaFeret = max(dimA, dimB)
                        if (dimA and dimB) != 0:
                            Aspect_Ratio = max(dimB, dimA) / min(dimA, dimB)
                        else:
                            Aspect_Ratio = 0
                        Length = min(dimA, dimB) * um_pix
                        Width = max(dimA, dimB) * um_pix

                        ellipse = cv2.fitEllipse(c)
                        (x, y), (major_axis, minor_axis), angle = ellipse
                        
                        if major_axis > minor_axis:
                            a = major_axis / 2.0
                            b = minor_axis / 2.0
                        else:
                            a = minor_axis / 2.0
                            b = major_axis / 2.0
                        eccentricity = np.sqrt(1 - (b**2 / a**2))
    
                        major_axis_length = major_axis / pixelsPerMetric * um_pix
                        minor_axis_length = minor_axis / pixelsPerMetric * um_pix

                        if dataset_name != 'hw_patterns':
                            CircularED = np.sqrt(4 * area / np.pi) * um_pix
                            Chords = cv2.arcLength(c, True) * um_pix
                            Roundness = 1 / Aspect_Ratio if Aspect_Ratio != 0 else 0
                            Sphericity = (2 * np.sqrt(np.pi * dimArea)) / dimPerimeter * um_pix
                            Circularity = 4 * np.pi * (dimArea / (dimPerimeter) ** 2) * um_pix
                            Feret_diam = diaFeret * um_pix

                            csvwriter.writerow([Length, Width, CircularED, Aspect_Ratio, Circularity, Chords, Feret_diam, Roundness, Sphericity, psum, test_img])
                        else:
                            mask = np.zeros(im.shape[:2], dtype=np.uint8)
                            cv2.drawContours(mask, [c], -1, 255, -1)
                            masked_image = cv2.bitwise_and(im, im, mask=mask)
                            
                            wavelengths = []
                            
                            for i in range(masked_image.shape[0]):
                                for j in range(masked_image.shape[1]):
                                    if mask[i, j] == 255:
                                        b, g, r = masked_image[i, j]
                                        wavelength = rgb_to_wavelength(b, g, r)
                                        wavelengths.append(wavelength)
                            
                            avg_velocity = sum(wavelengths) / len(wavelengths)


                            # Compute velocities within the mask
                            # flow_vectors = detect_arrows(masked_image)
                            # avg_direction = np.mean(flow_vectors, axis=0)

                            flow_vectors = detect_arrows(masked_image)
                            if flow_vectors:
                                avg_direction = np.mean(flow_vectors, axis=0)
                            else:
                                avg_direction = (0, 0)

                            avg_direction_x, avg_direction_y = avg_direction[0], avg_direction[1]
                            
                            # if flow_vectors:
                            #     avg_direction = np.mean(flow_vectors, axis=0)
                            # else:
                            #     avg_direction = (0, 0)

                            csvwriter.writerow([Length, Width, major_axis_length, minor_axis_length, eccentricity, global_min_wavelength, avg_velocity, global_max_wavelength, avg_direction_x, avg_direction_y, test_img])
