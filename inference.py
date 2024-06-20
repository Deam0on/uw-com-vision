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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65

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

def run_inference(dataset_name, output_dir, visualize=False):
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
    predictor = choose_and_use_model(trained_model_paths, selected_model_dataset)
    
    metadata = MetadataCatalog.get(f"{dataset_name}_train")
    
    image_folder_path = get_image_folder_path()
    
    # Path to save outputs
    path = output_dir
    os.makedirs(path, exist_ok=True)
    inpath = image_folder_path
    images_name = [f for f in os.listdir(inpath) if f.endswith('.tif')]
    
    Img_ID = []
    EncodedPixels = []

    test_count = 0
        
    conv = lambda l: ' '.join(map(str, l))
    
    for name in images_name:
        image = cv2.imread(os.path.join(inpath, name))
        outputs = predictor(image)
        masks = postprocess_masks(
            np.asarray(outputs["instances"].to('cpu')._fields['pred_masks']),
            outputs["instances"].to('cpu')._fields['scores'].numpy(), image)
    
        if masks:  # If any objects are detected in this image
            for i in range(len(masks)):  # Loop all instances
                Img_ID.append(name.replace('.tif', ''))
                EncodedPixels.append(conv(rle_encoding(masks[i])))
    
    # Save inference results
    df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
    df.to_csv(os.path.join(path, "R50_flip_results.csv"), index=False, sep=',')

    for x_pred in [0, 1]:
        TList = []
        PList = []
        csv_filename = f'results_x_pred_{x_pred}.csv'
        test_img_path = image_folder_path
    
        # Open CSV file before processing images
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['length', 'width', 'circularED', 'aspectRatio', 'circularity', 'chords', 'ferret', 'round', 'sphere', 'psum', 'name'])
    
            for test_img in os.listdir(test_img_path):
                input_path = os.path.join(test_img_path, test_img)
                im = cv2.imread(input_path)
    
                # Convert image to grayscale
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
                # Use canny edge detection
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)

                if dataset_name != 'hw_patterns':
                    # Execute this block for datasets other than 'hw_patterns'
                    reader = easyocr.Reader(['en'])
                    result = reader.readtext(gray, detail=0, paragraph=False, contrast_ths=0.85, adjust_contrast=0.85, add_margin=0.25, width_ths=0.25, decoder='beamsearch')
                    if result:  # Ensure result is not empty
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
                    # Placeholder: You can add any specific handling for 'hw_patterns' here if needed.
                    um_pix = 1
                    psum = '0'
    
                GetInference(predictor, im, x_pred, metadata, test_img)  # Ensure this function is correctly defined elsewhere
                GetCounts(predictor, im, TList, PList)  # Ensure this function is correctly defined elsewhere
    
                outputs = predictor(im)
                inst_out = outputs['instances']
                filtered_instances = inst_out[inst_out.pred_classes == x_pred]
                mask_array = filtered_instances.pred_masks.to("cpu").numpy()
                num_instances = mask_array.shape[0]
                mask_array = np.moveaxis(mask_array, 0, -1)
                output = np.zeros_like(im)

                for i in range(num_instances):
                    # Initialize a new output array for each mask
                    single_output = np.zeros_like(output)
                    mask = mask_array[:, :, i:(i + 1)]
                    single_output = np.where(mask == True, 255, single_output)
                    
                    # Save each mask image separately for debugging
                    mask_filename = os.path.join(output_dir, f'mask_{i}.jpg')
                    cv2.imwrite(mask_filename, single_output)
                    
                    # Convert the single mask to grayscale for contour detection
                    single_im_mask = cv2.cvtColor(single_output, cv2.COLOR_BGR2GRAY)
                    single_cnts = cv2.findContours(single_im_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    single_cnts = imutils.grab_contours(single_cnts)
                
                    for c in single_cnts:
                        pixelsPerMetric = 1  # or 0.85, correction
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

                        if dataset_name != 'hw_patterns':
                            dimArea = area / pixelsPerMetric
                            dimPerimeter = perimeter / pixelsPerMetric
                            diaFeret = max(dimA, dimB)
                            # Execute this block for datasets other than 'hw_patterns'
                            if (dimA and dimB) != 0:
                                Aspect_Ratio = max(dimB, dimA) / min(dimA, dimB)
                            else:
                                Aspect_Ratio = 0
                            Length = min(dimA, dimB) * um_pix
                            Width = max(dimA, dimB) * um_pix
                            CircularED = np.sqrt(4 * area / np.pi) * um_pix
                            Chords = cv2.arcLength(c, True) * um_pix
                            Roundness = 1 / Aspect_Ratio if Aspect_Ratio != 0 else 0
                            Sphericity = (2 * np.sqrt(np.pi * dimArea)) / dimPerimeter * um_pix
                            Circularity = 4 * np.pi * (dimArea / (dimPerimeter) ** 2) * um_pix
                            Feret_diam = diaFeret * um_pix

                            csvwriter.writerow([Length, Width, CircularED, Aspect_Ratio, Circularity, Chords, Feret_diam, Roundness, Sphericity, psum, test_img])
                        else:
                            dimArea = area / pixelsPerMetric
                            dimPerimeter = perimeter / pixelsPerMetric
                            diaFeret = max(dimA, dimB)
                            # Execute this block for 'hw_patterns'
                            if (dimA and dimB) != 0:
                                Aspect_Ratio = max(dimB, dimA) / min(dimA, dimB)
                            else:
                                Aspect_Ratio = 0
                            Length = min(dimA, dimB)
                            Width = max(dimA, dimB)

                            csvwriter.writerow([Length, Width, test_img])
