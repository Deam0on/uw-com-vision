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
:
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

    Parameters:
    - predictor: The predictor object used for inference.
    - im: The image to perform inference on.
    - x_pred: The class to filter predicted instances by.
    - metadata: Metadata for visualization.
    - test_img: Path to save the test image.

    Returns:
    - None
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

    Parameters:
    - predictor: The predictor object used for inference.
    - im: The image to perform inference on.
    - TList: List to store counts of the first class.
    - PList: List to store counts of the second class.

    Returns:
    - None
    """
    outputs = predictor(im)
    classes = outputs["instances"].pred_classes.to("cpu").numpy()
    TotalCount = sum(classes == 0) + sum(classes == 1)
    TCount = sum(classes == 0)
    PCount = sum(classes == 1)
    TList.append(TCount)
    PList.append(PCount)

def rgb_to_hsv(r, g, b):
    """
    Converts RGB color values to HSV color values.

    Parameters:
    - r: Red component.
    - g: Green component.
    - b: Blue component.

    Returns:
    - tuple: HSV color values.
    """
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

def hue_to_wavelength(hue):
    """
    Converts hue value to wavelength.

    Parameters:
    - hue: Hue value.

    Returns:
    - float: Wavelength in nanometers.
    """
    assert hue >= 0
    assert hue <= 270

    wavelength = 620 - 170 / 270 * hue
    return wavelength

def rgb_to_wavelength(r, g, b):
    """
    Converts RGB color values to wavelength.

    Parameters:
    - r: Red component.
    - g: Green component.
    - b: Blue component.

    Returns:
    - float: Wavelength in nanometers.
    """
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
        

        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Calculate the direction vector
        direction = (vx[0], vy[0])
        flow_vectors.append(direction)
    
    return flow_vectors

def run_inference(dataset_name, output_dir, visualize=False, threshold=0.65):
    """
    Runs inference on images in the specified directory using the provided model.

    Parameters:
    - dataset_name: Name of the dataset.
    - output_dir: Directory to save inference results.
    - visualize: Boolean, if True, save visualizations of predictions.
    - threshold: Threshold for model prediction scores.

    Returns:
    - None
    """
    dataset_info = read_dataset_info('/home/deamoon_uw_nn/uw-com-vision/dataset_info.json')
    register_datasets(dataset_info, dataset_name)
    
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
                csvwriter.writerow(['E_major', 'E_minor', 'Eccentricity', 'D10_avg_velocity', 'avg_velocity', 'D90_avg_velocity', 'avg_direction_x', 'avg_direction_y', 'magnitude', 'angle', 'angle_degrees', 'name'])

    
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
                            
                            wavelengths = sorted(wavelengths)
                            D10 = np.percentile(wavelengths, 10)
                            D90 = np.percentile(wavelengths, 90)
                            
                            avg_velocity = ((sum(wavelengths) / len(wavelengths)) - global_min_wavelength) / (global_max_wavelength - global_min_wavelength)
                            normalized_D10 = (D10 - global_min_wavelength) / (global_max_wavelength - global_min_wavelength)
                            normalized_D90 = (D90 - global_min_wavelength) / (global_max_wavelength - global_min_wavelength)

                            flow_vectors = detect_arrows(masked_image)
                            if flow_vectors:
                                avg_direction = np.mean(flow_vectors, axis=0)
                            else:
                                avg_direction = (0, 0)

                            avg_direction_x, avg_direction_y = avg_direction[0], avg_direction[1]
                            magnitude = math.sqrt(avg_direction_x**2 + avg_direction_y**2)
                            
                            angle = math.atan2(avg_direction_y, avg_direction_x)
                            angle_degrees = math.degrees(angle)
                            
                            csvwriter.writerow([major_axis_length, minor_axis_length, eccentricity, normalized_D10, avg_velocity, normalized_D90, avg_direction_x, avg_direction_y, magnitude, angle, angle_degrees, test_img])
