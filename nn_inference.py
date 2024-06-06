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
import optuna
import easyocr
import re
from numpy import sqrt

## Def for dataset build, SA annotated data, SA format, WARNING, NO POLYLINES
def get_superannotate_dicts(img_dir, label_dir):
    dataset_dicts = []
    idx = 0
    for r, d, f in os.walk(label_dir):
        for file in f:
            if file.endswith(".json"):
                json_file = os.path.join(r, file)

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
                        # Create a circle of radius 1 around the centre point:
                        circ = shapely.geometry.Point(ellipse[0]).buffer(1)
                        # Create ellipse along x and y:
                        ell = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
                        # rotate the ellipse(clockwise, x axis pointing right):
                        ellr = shapely.affinity.rotate(ell, ellipse[2])

                        px, py = ellr.exterior.coords.xy

                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py) ]
                        poly = [p for x in poly for p in x]
                        
                    elif type == "polygon":
                        px = anno["points"][0:-1:2]  #0 -1 2
                        py = anno["points"][1:-1:2] # 1 -1 2
                        px.append(anno["points"][0])    # 0
                        py.append(anno["points"][-1])   # -1

                        poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py) ]
                        poly = [p for x in poly for p in x]

                    if "throat" in categoryName :
                        category_id = 0
                    elif "pore" in categoryName :
                        category_id = 1
                    else:
                        raise ValueError("Category Name Not Found: "+ categoryName)

                    obj = {
                        "bbox":[np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": category_id,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts

## Def custom mapper, rand changes to dataset imgs, induce variability to dataset
def custom_mapper(dataset_dict):
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
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
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
  
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


## Replace default trainer, subs 2x 1D method
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

#Dataset load
keywords = ["Train", "Test"]
for d in keywords:
    #DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("dataset/multiclass/" + d, "dataset/multiclass/train/*.json"))
    DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("/home/deamoon_uw_nn/DATASET/" + d + "/", 
                                                                                   "/home/deamoon_uw_nn/DATASET/" + d + "/"))
    MetadataCatalog.get("multiclass_Train").set( thing_classes=["throat","pore"])
  
multiclass_metadata = MetadataCatalog.get("multiclass_Train").set( thing_classes=["throat","pore"])
multiclass_test_metadata = MetadataCatalog.get("multiclass_Test").set( thing_classes=["throat","pore"])


def get_image_folder_path(base_path='/home/deamoon_uw_nn/DATASET/INFERENCE'):
    inference_path = os.path.join(base_path)
    upload_path = os.path.join(base_path, 'UPLOAD')

    if os.path.exists(inference_path) and any(os.path.isfile(os.path.join(inference_path, f)) for f in os.listdir(inference_path)):
        return inference_path
    elif os.path.exists(upload_path) and any(os.path.isfile(os.path.join(upload_path, f)) for f in os.listdir(upload_path)):
        return upload_path
    else:
        raise FileNotFoundError("No images found in INFERENCE or INFERENCE/UPLOAD folders.")

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
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

cfg.MODEL.DEVICE = "cuda"
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

def rle_encode(img):
    pixels = img.flatten()
    if not np.any(pixels):
        return ''
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def postprocess_masks(ori_mask, ori_score, image, min_crys_size=2):
    height, width = image.shape[:2]

    if len(ori_mask) == 0 or np.max(ori_score) < 0.5:
        return []

    valid_masks = [ori_mask[i] for i in range(len(ori_mask)) if np.sum(ori_mask[i]) > min_crys_size]
    if not valid_masks:
        return []

    masks = []
    overlap = np.zeros([height, width])

    for mask in valid_masks:
        mask = binary_fill_holes(mask).astype(np.uint8)
        mask = erosion(dilation(mask))
        overlap += mask
        mask[overlap > 1] = 0
        if np.max(label(mask)) <= 1:
            masks.append(mask)
    return masks


# path = "./output/"  # the weight save path
# inpath = image_folder_path
# images_name = listdir(inpath)
# images_name = [f for f in os.listdir(inpath) if f.endswith('.tif')]

Img_ID = []
EncodedPixels = []
num = 0
conv = lambda l: ' '.join(map(str, l))

path = "./output/"
inpath = image_folder_path
images_name = [f for f in os.listdir(inpath) if f.endswith('.tif')]

for name in images_name:
    image = cv2.imread(os.path.join(inpath, name))
    if image is None:
        print(f"Error loading image {name}, skipping...")
        continue
    outputs = predictor(image)
    masks = postprocess_masks(
        np.asarray(outputs["instances"].to('cpu')._fields['pred_masks']),
        outputs["instances"].to('cpu')._fields['scores'].numpy(),
        image
    )
    if masks:
        for i, mask in enumerate(masks):
            Img_ID.append(name.replace('.tif', ''))
            EncodedPixels.append(rle_encode(mask))

## save inference 
df = pd.DataFrame({"ImageId": Img_ID, "EncodedPixels": EncodedPixels})
df.to_csv("./output/R50_flip_" + ".csv", index=False, sep=',')

## def for analysis and measurements
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

## sub inference from mask
def GetInference():
  outputs = predictor(im)

  # Get all instances
  inst_out = outputs['instances']

  # Filter instances where predicted class is 3
  filtered_instances = inst_out[inst_out.pred_classes == x_pred]
    
  v = Visualizer(im[:, :, ::-1],
                  metadata=multiclass_test_metadata,
                  scale=1,
                  instance_mode=ColorMode.SEGMENTATION)
  out = v.draw_instance_predictions(filtered_instances.to("cpu"))  
  # v.save("test.png")
  cv2.imwrite(test_img + '_' + str(x_pred) + "__pred.png",out.get_image()[:, :, ::-1])

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
          
for x_pred in [0, 1]:
    csv_filename = f'results_x_pred_{x_pred}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['length', 'width', 'circularED', 'aspectRatio', 'circularity', 'chords', 'ferret', 'round', 'sphere', 'psum', 'name'])
    
        for test_img in os.listdir(test_img_path):
            input_path = os.path.join(test_img_path, test_img)
            im = cv2.imread(input_path)
            if im is None:
                print(f"Error loading image {test_img}, skipping...")
                continue
            source_image_filename = test_img
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            reader = easyocr.Reader(['en'])
            result = reader.readtext(gray, detail=0, paragraph=False)
            psum = re.sub("[^0-9]", "", result[0] if result else "1")
            lines_list = []
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=1)
            
            for points in lines:
                x1, y1, x2, y2 = points[0]
                cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                scale_len = sqrt((x2 - x1)**2 + (y2 - y1)**2)
                um_pix = float(psum) / scale_len

            GetInference()
            GetCounts()

            outputs = predictor(im)
            filtered_instances = outputs['instances'][outputs['instances'].pred_classes == x_pred]
            mask_array = filtered_instances.pred_masks.to("cpu").numpy()

            num_instances = mask_array.shape[0]
            mask_array = np.moveaxis(mask_array, 0, -1)
            output = np.zeros_like(im)
            for i in range(num_instances):
                output = np.where(mask_array[:, :, i] == True, 255, output)
            im_mask = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(im_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            if cnts:
                (cnts, _) = contours.sort_contours(cnts)
                pixelsPerMetric = None
                for c in cnts:
                    if cv2.contourArea(c) < 100:
                        continue
                    area = cv2.contourArea(c)
                    perimeter = cv2.arcLength(c, True)
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
                    box = perspective.order_points(box)
                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    if pixelsPerMetric is None:
                        pixelsPerMetric = dB / max(dA, dB)
                    dimA = dA / pixelsPerMetric
                    dimB = dB / pixelsPerMetric
                    Aspect_Ratio = max(dimA, dimB) / min(dimA, dimB) if min(dimA, dimB) != 0 else 0
                    Length = min(dimA, dimB) * um_pix
                    Width = max(dimA, dimB) * um_pix
                    CircularED = np.sqrt(4 * area / np.pi) * um_pix
                    Chords = perimeter * um_pix
                    Roundness = 1 / Aspect_Ratio if Aspect_Ratio != 0 else 0
                    Sphericity = (2 * np.sqrt(np.pi * area / pixelsPerMetric)) / perimeter * um_pix
                    Circularity = 4 * np.pi * (area / (perimeter ** 2)) * um_pix
                    Feret_diam = max(dimA, dimB) * um_pix

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
