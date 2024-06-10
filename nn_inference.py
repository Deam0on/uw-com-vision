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
from imutils import perspective
from imutils import contours
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
import copy
from shapely.geometry import Point
from shapely.affinity import scale, rotate
import matplotlib.pyplot as plt
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import pycocotools.mask as mask_util
from skimage.measure import find_contours
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, erosion
from google.cloud import storage

import easyocr
import re
from numpy import sqrt


def register_datasets(dataset_info, test_size=0.2):
    for dataset_name, info in dataset_info.items():
        img_dir, label_dir, thing_classes = info

        # Load or split the dataset
        # split_dir = os.path.join(label_dir, 'splits')
        split_dir = "/home/deamoon_uw_nn/split_dir/"
        split_file = os.path.join(split_dir, f"{dataset_name}_split.json")
        
        # if os.path.exists(split_file):
        #     with open(split_file, 'r') as f:
        #         split_data = json.load(f)
        #     train_files = split_data['train']
        #     test_files = split_data['test']
        # else:
        #     train_files, test_files = split_dataset(img_dir, label_dir, dataset_name, test_size)

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            train_files = split_data['train']
            test_files = split_data['test']
        else:
            print("No split training data found!")

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

# Utility function to handle the split dictionaries
def get_split_dicts(img_dir, label_dir, files):
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
        idx = idx + 1
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

def custom_mapper(dataset_dicts):
    dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
    image = utils.read_image(dataset_dicts["file_name"], format="BGR")
    transform_list = [
        T.Resize((800,800)),
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

## Replace default trainer, subs 2x 1D method
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

# Example dataset info with specific classes
dataset_info = {
    "polyhipes": ("/home/deamoon_uw_nn/DATASET/polyhipes/", "/home/deamoon_uw_nn/DATASET/polyhipes/", ["throat", "pore"])
}

register_datasets(dataset_info)



# def split_dataset(img_dir, label_dir, test_size=0.2, seed=42):
#     random.seed(seed)
#     label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
#     train_files, test_files = train_test_split(label_files, test_size=test_size, random_state=seed)

#     return train_files, test_files

# def register_datasets(dataset_info, test_size=0.2):
#     for dataset_name, info in dataset_info.items():
#         img_dir, label_dir, thing_classes = info

#         # Split the dataset
#         train_files, test_files = split_dataset(img_dir, label_dir, test_size)

#         # Register training dataset
#         DatasetCatalog.register(
#             f"{dataset_name}_train",
#             lambda d=dataset_name, img_dir=img_dir, label_dir=label_dir, files=train_files:
#             get_split_dicts(img_dir, label_dir, files)
#         )
#         MetadataCatalog.get(f"{dataset_name}_train").set(thing_classes=thing_classes)

#         # Register testing dataset
#         DatasetCatalog.register(
#             f"{dataset_name}_test",
#             lambda d=dataset_name, img_dir=img_dir, label_dir=label_dir, files=test_files:
#             get_split_dicts(img_dir, label_dir, files)
#         )
#         MetadataCatalog.get(f"{dataset_name}_test").set(thing_classes=thing_classes)

# # Utility function to handle the split dictionaries
# def get_split_dicts(img_dir, label_dir, files):
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
#         idx = idx + 1
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

# def custom_mapper(dataset_dicts):
#     dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
#     image = utils.read_image(dataset_dicts["file_name"], format="BGR")
#     transform_list = [
#         T.Resize((800,800)),
#         T.RandomBrightness(0.8, 1.8),
#         T.RandomContrast(0.6, 1.3),
#         T.RandomSaturation(0.8, 1.4),
#         T.RandomRotation(angle=[90, 90]),
#         T.RandomLighting(0.7),
#         T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
#     ]

#     image, transforms = T.apply_transform_gens(transform_list, image)
#     dataset_dicts["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dicts.pop("annotations")
#         if obj.get("iscrowd", 0) == 0
#     ]

#     instances = utils.annotations_to_instances(annos, image.shape[:2])
#     dataset_dicts["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dicts

# ## Replace default trainer, subs 2x 1D method
# class CustomTrainer(DefaultTrainer):
#     @classmethod
#     def build_train_loader(cls, cfg):
#         return build_detection_train_loader(cfg, mapper=custom_mapper)

# # Example dataset info with specific classes
# dataset_info = {
#     "polyhipes": ("/home/deamoon_uw_nn/DATASET/polyhipes/", "/home/deamoon_uw_nn/DATASET/polyhipes/", ["throat", "pore"])
# }

# register_datasets(dataset_info)
# def load_model(cfg, model_path):
#     cfg.MODEL.WEIGHTS = model_path
#     predictor = DefaultPredictor(cfg)
#     return predictor

def get_trained_model_paths(base_dir):
    model_paths = {}
    for dataset_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, dataset_name)
        model_path = os.path.join(model_dir, "model_final.pth")
        if os.path.exists(model_path):
            model_paths[dataset_name] = model_path
    return model_paths

def load_model(cfg, model_path, dataset_name):
    cfg.MODEL.WEIGHTS = model_path
    thing_classes = MetadataCatalog.get(f"{dataset_name}_train").thing_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    predictor = DefaultPredictor(cfg)
    return predictor

def choose_and_use_model(model_paths, dataset_name):
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

# Example usage:
trained_model_paths = get_trained_model_paths("./trained_models")
selected_model_dataset = "polyhipes"  # User-selected model
predictor = choose_and_use_model(trained_model_paths, selected_model_dataset)

metadata = MetadataCatalog.get(f"{selected_model_dataset}_train")

def get_image_folder_path(base_path='/home/deamoon_uw_nn/DATASET/INFERENCE'):
    """
    This function checks whether the images are in the base folder or in the UPLOAD subfolder.
    It returns the path to the folder containing the images.

    Parameters:
    base_path (str): The base path where the INFERENCE folder is located.

    Returns:
    str: The path to the folder containing the images.
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

    # If no images found in either folder, return None or raise an exception
    else:
        raise FileNotFoundError("No images found in INFERENCE or INFERENCE/UPLOAD folders.")

# Example usage:
# Assuming you are running the script from the same directory where the INFERENCE folder is located
image_folder_path = get_image_folder_path()

## Collect prediction masks
# Convert binary_mask to RLE
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(
        itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            d( )
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

THRESHOLDS = [.18, .35, .58]
MIN_PIXELS = [75, 150, 75]

# encode RLE
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# get mask as bin
def get_masks(fn, predictor):
    im = cv2.imread(fn)
    pred = predictor(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= THRESHOLDS[pred_class]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    for mask in pred_masks:
        mask = mask * (1-used)
        # Skip predictions with small area
        if mask.sum() >= MIN_PIXELS[pred_class]:
            used += mask
            res.append(rle_encode(mask))
    return res

# inference with trained model
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45   # set the testing threshold for this model
# predictor = DefaultPredictor(cfg)

# cfg.MODEL.DEVICE = "cuda"
MetadataCatalog.get("multiclass_Train").set(
         things_classes=["throat","pore"])
MetadataCatalog.get("multiclass_Train").set(
         things_colors=[(146, 19, 26), (47, 213, 218)])
multiclass_test_metadata = MetadataCatalog.get("multiclass_Train")

### Conversion from RLE to BitMask
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
    for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape) # Needed to align to RLE direction ##

def rle_encoding(x):
    # .T sets Fortran order down-then-right
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def postprocess_masks(ori_mask, ori_score, image, min_crys_size=2):

    """Clean overlaps between bounding boxes, """
    """ fill small holes, smooth boundaries"""
    image = image[:, :,::-1]
    height, width = image.shape[:2]

    score_threshold = 0.5

    if len(ori_mask) == 0 or ori_score.all() < score_threshold:
        return

    keep_ind = np.where(np.sum(ori_mask, axis=(0, 1)) > min_crys_size)[0]
    if len(keep_ind) < len(ori_mask):  # keep_ind possible to be zero zero
        if(keep_ind.shape[0] != 0):
            ori_mask = ori_mask[:keep_ind.shape[0]]  # shape[0]:to int
            ori_score = ori_score[:keep_ind.shape[0]]
        else:
            ori_mask = []
            ori_score = []

    overlap = np.zeros([height, width])

    masks = []

    # Removes overlaps from masks with lower score
    for i in range(len(ori_mask)):
        # Fill holes inside the mask
        mask = binary_fill_holes(ori_mask[i]).astype(np.uint8)
        # Smoothen edges using dilation and erosion
        mask = erosion(dilation(mask))
        # Delete overlaps
        overlap += mask
        mask[overlap > 1] = 0
        out_label = label(mask)
        # Remove all the pieces if there are more than one pieces
        if out_label.max() > 1:
            mask[()] = 0

        masks.append(mask)

    return masks

path = "./output/"  # the weight save path
inpath = image_folder_path
images_name = listdir(inpath)
images_name = [f for f in os.listdir(inpath) if f.endswith('.tif')]

Img_ID = []
EncodedPixels = []
num = 0
conv = lambda l: ' '.join(map(str, l))

for name in images_name:
    image = cv2.imread(inpath + "/" + name)
    outputs = predictor(image)
    num += 1
    # print(np.asarray(outputs["instances"].to('cpu')._fields['pred_masks'][0]).shape)
    masks = postprocess_masks(
        np.asarray(outputs["instances"].to('cpu')._fields['pred_masks']),
            outputs["instances"].to('cpu')._fields['scores'].numpy(), image)

    if masks:  # If any objects are detected in this image
            for i in range(len(masks)):  # Loop all instances
                Img_ID.append(name.replace('.tif', ''))
                EncodedPixels.append(conv(rle_encoding(masks[i])))

## save inference
df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
df.to_csv("./output/R50_flip_" + ".csv", index=False, sep=',')

## def for analysis and measurements
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

## sub inference from mask
# def GetInference():
#   outputs = predictor(im)

#   # Get all instances
#   inst_out = outputs['instances']

#   # Filter instances where predicted class is 3
#   filtered_instances = inst_out[inst_out.pred_classes == x_pred]
#   thing_classes = MetadataCatalog.get(f"{selected_model_dataset}_train").thing_classes
#   v = Visualizer(im[:, :, ::-1],
#                   metadata=thing_classes,
#                   scale=1,
#                   instance_mode=ColorMode.SEGMENTATION)
#   out = v.draw_instance_predictions(filtered_instances.to("cpu"))
#   # v.save("test.png")
#   cv2.imwrite(test_img + '_' + str(x_pred) + "__pred.png",out.get_image()[:, :, ::-1])

## sub inference from mask
# def GetInference():
#   outputs = predictor(im)

#   # Get all instances
#   inst_out = outputs['instances']

#   # Filter instances where predicted class is 3
#   filtered_instances = inst_out[inst_out.pred_classes == x_pred]
    
#   v = Visualizer(im[:, :, ::-1],
#                   metadata=multiclass_test_metadata,
#                   scale=1,
#                   instance_mode=ColorMode.SEGMENTATION)
#   out = v.draw_instance_predictions(filtered_instances.to("cpu"))  
#   # v.save("test.png")
#   cv2.imwrite(test_img + '_' + str(x_pred) + "__pred.png",out.get_image()[:, :, ::-1])

def GetInference():
    outputs = predictor(im)

    # Ensure the outputs contain 'instances'
    if 'instances' not in outputs:
        raise ValueError("No 'instances' found in outputs from the model.")

    instances = outputs['instances']
    
    if not isinstance(instances, detectron2.structures.Instances):
        raise TypeError(f"Expected 'Instances', got {type(instances)}.")

    # Filter instances by class
    filtered_instances = instances[instances.pred_classes == x_pred]
    
    # Ensure metadata is a Metadata object
    if not hasattr(metadata, 'get'):
        raise TypeError(f"Expected metadata to have 'get' method, got {type(metadata)} instead.")

    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=1,
                   instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(filtered_instances.to("cpu"))

    # Save the output image
    output_image_path = os.path.join(output_dir, f"pred_{x_pred}.png")
    cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
    print(f"Saved inference image for class {x_pred} to {output_image_path}")



## count types

## count types
def GetCounts():
  outputs = predictor(im)
  classes = outputs["instances"].pred_classes.to("cpu").numpy()
  TotalCount = sum(classes==0)+sum(classes==1)
  TCount = sum(classes==0)
  PCount = sum(classes==1)
  TList.append(TCount)
  PList.append(PCount)

## get mask contours for outlines / ferret
# def GetMask_Contours():
          
for x_pred in [0,1]:

    ## create and append lists
    # lengthList = list()
    # widthList = list()
    # circularEDList = list()
    # aspectRatioList = list()
    # circularityList = list()
    # chordsList = list()
    # ferretList = list()
    # roundList = list()
    # sphereList = list()
    TList = list()
    PList = list()
    # psum_list = list()
    # name_list = list()
    tT = 0
    tP = 0
    count = 0    
    csv_filename = f'results_x_pred_{x_pred}.csv'
    test_img_path = image_folder_path
    x_th = len(test_img_path)
    x_c = 0



    # Open CSV file before processing images
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header row (adjust this as per your data)
        csvwriter.writerow(['length', 'width', 'circularED', 'aspectRatio', 'circularity', 'chords', 'ferret', 'round', 'sphere', 'psum', 'name'])
    
        for test_img in os.listdir(test_img_path):
    
            # Write the measurements and descriptors to the CSV file
            # classes_of_interest = [keywds.index(k)]
            input_path = os.path.join(test_img_path, test_img)
            im = cv2.imread(input_path)
            source_image_filename = test_img
        
            count = count+1
        
            # Convert image to grayscale
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            
            # Use canny edge detection
            edges = cv2.Canny(gray,50,150,apertureSize=3)
            
            reader = easyocr.Reader(['en'])
            result = reader.readtext(gray, 
                         detail=0,             # Return only the text
                         paragraph=False,       # Treat each line separately
                         contrast_ths=0.85,      # Increase contrast threshold
                         adjust_contrast=0.85,   # Adjust contrast
                         add_margin=0.25,        # Add margin around text
                         width_ths=0.25,         # Width threshold for text boxes
                         decoder='beamsearch')  # Use beamsearch for decoding
            pxum_r = result[0]
            psum = re.sub("[^0-9]", "", pxum_r)

            # reader = easyocr.Reader(['en'])
            # result = reader.readtext(gray, detail = 0)
            # pxum_r = result[0]
            # psum = re.sub("[^0-9]", "", pxum_r)
        
            lines_list =[]
            lines = cv2.HoughLinesP(
                        edges, # Input edge image
                        1, # Distance resolution in pixels
                        np.pi/180, # Angle resolution in radians
                        threshold=100, # Min number of votes for valid line
                        minLineLength=100, # Min allowed length of line
                        maxLineGap=1 # Max allowed gap between line for joining them
                        )
            
            # Iterate over points
            for points in lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]
                # Draw the lines joing the points
                # On the original image
                cv2.line(im,(x1,y1),(x2,y2),(0,255,0),2)
                # Maintain a simples lookup list for points
                lines_list.append([(x1,y1),(x2,y2)])
                scale_len = sqrt((x2-x1)**2+(y2-y1)**2)
                um_pix = float(psum)/scale_len    
            # um_pix = 1
        
            GetInference()
            GetCounts()

            outputs = predictor(im)
            
            # Get all instances
            inst_out = outputs['instances']
            
            # Filter instances where predicted class is 3
            filtered_instances = inst_out[inst_out.pred_classes == x_pred]
            
            # Now extract the masks for these filtered instances
            mask_array = filtered_instances.pred_masks.to("cpu").numpy()
            
            # instances = instances[instances.pred_classes == 3]
            # mask_array = outputs['instances'].pred_masks==3.to("cpu").numpy()
            num_instances = mask_array.shape[0]
            mask_array = np.moveaxis(mask_array, 0, -1)
            mask_array_instance = []
            output = np.zeros_like(im)
            fig = plt.figure(figsize=(15, 20))
            for i in range(num_instances):
                mask_array_instance.append(mask_array[:, :, i:(i+1)])
                output = np.where(mask_array_instance[i] == True, 255, output)
            imm = Image.fromarray(output)
            imm.save('predicted_masks.jpg')
            cv2.imwrite('Masks.jpg', output) #mask
            im_mask = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(im_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            if len(cnts) > 0:
              
                (cnts, _) = contours.sort_contours(cnts)
                pixelsPerMetric = 0.85
                
                for c in cnts:
                    if cv2.contourArea(c) < 100:
                        continue
                    area = cv2.contourArea(c)
                    perimeter = cv2.arcLength(c, True)
                    
                    orig = im_mask.copy()
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
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
                    if pixelsPerMetric is None:
                        pixelsPerMetric = dB / width
                    dimA = dA / pixelsPerMetric
                    dimB = dB / pixelsPerMetric
                    dimArea = area/pixelsPerMetric
                    dimPerimeter = perimeter/pixelsPerMetric
                    diaFeret = max(dimA, dimB)
                    if (dimA and dimB) !=0:
                        Aspect_Ratio = max(dimB,dimA)/min(dimA,dimB)
                    else:
                        Aspect_Ratio = 0
                    Length = min(dimA, dimB)*um_pix
                    Width = max(dimA, dimB)*um_pix
                    CircularED = np.sqrt(4*area/np.pi)*um_pix
                    Chords = cv2.arcLength(c,True)*um_pix
                    Roundness = 1/(Aspect_Ratio) if Aspect_Ratio != 0 else 0
                    Sphericity = (2*np.sqrt(np.pi*dimArea))/dimPerimeter*um_pix
                    Circularity = 4*np.pi*(dimArea/(dimPerimeter)**2)*um_pix
                    Feret_diam = diaFeret*um_pix
    
                    csvwriter.writerow([Length, Width, CircularED, Aspect_Ratio, Circularity, Chords, Feret_diam, Roundness, Sphericity, psum, test_img])
    
            # Create a CSV file at the end of each iteration of the outer loop
            # csv_filename = f'results_x_pred_{x_pred}.csv'
            # with open(csv_filename, 'a', newline='') as csvfile:
            #     csvwriter = csv.writer(csvfile)
            #     # Write the header
            #     csvwriter.writerow(['length', 'width', 'circularED', 'aspectRatio', 'circularity', 'chords', 'ferret', 'round', 'sphere', 'psum', 'name'])
            #     # Write the data rows
            #     for row in zip(lengthList, widthList, circularEDList, aspectRatioList, circularityList, chordsList, ferretList, roundList, sphereList, psum_list, name_list):
            #         csvwriter.writerow(row)
        
            # print(f'Data for x_pred={x_pred} written to {csv_filename}')
    
    
    for T in range(0, len(TList)):
        tT = tT + TList[T]
    for P in range(0, len(PList)):
        tP = tP + PList[P]
    
    print("No. (Total) of Pores:  " + repr(tP))
    print("No. (Total) of Pore Throats:  " + repr(tT))
