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

## Def for dataset build, SA annotated data, SA format, WARNING, NO POLYLINES
def get_superannotate_dicts(img_dir, label_dir):
    dataset_dicts = []
    idx = 0
    for r, d, f in os.walk(label_dir):
        for file in f:
            if file.endswith(".json"):
                json_file = os.path.join(r, file)
                print(json_file)

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
                    elif type == "polygon":
                        px = anno["points"][0:-1:2]  #0 -1 2
                        py = anno["points"][1:-1:2] # 1 -1 2
                        px.append(anno["points"][0])    # 0
                        py.append(anno["points"][-1])   # -1
                      
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py) ]
                    poly = [p for x in poly for p in x]

                    if "Scale bar" in categoryName :
                        category_id = 0
                    elif "Wall thickness of polyHIPEs" in categoryName :
                        category_id = 1
                    elif "Pore throats of polyHIPEs" in categoryName :
                        category_id = 2
                    elif "Pores of polyHIPEs" in categoryName :
                        category_id = 3
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

## Load custom dataset, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE THING CLASSES TO LOAD FROM FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Dataset load
keywords = ["Train", "Test"]
for d in keywords:
    #DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("dataset/multiclass/" + d, "dataset/multiclass/train/*.json"))
    DatasetCatalog.register("multiclass_" + d, lambda d=d: get_superannotate_dicts("/home/deamoon_uw_nn/DATASET/" + d + "/", 
                                                                                   "/home/deamoon_uw_nn/DATASET/" + d + "/"))
    MetadataCatalog.get("multiclass_Train").set( thing_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])
  
multiclass_metadata = MetadataCatalog.get("multiclass_Train").set( thing_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])
multiclass_test_metadata = MetadataCatalog.get("multiclass_Test").set( thing_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])

## Def det2 hyperparameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO OPTUNA OPTIMIZATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("multiclass_Train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.DEVICE = "cuda"

## Train model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

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

# REDO catalog, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!things classes from file, as above
cfg.MODEL.DEVICE = "cuda"
MetadataCatalog.get("multiclass_Train").set(
         things_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])
MetadataCatalog.get("multiclass_Train").set(
         things_colors=[(115, 254, 248), (239, 254, 21), (146, 19, 26), (47, 213, 218)])
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
inpath = "/home/deamoon_uw_nn/DATASET/INFERENCE/"
images_name = listdir(inpath)
images_name = [f for f in os.listdir(inpath) if f.endswith('.tif')]
print(images_name)

Img_ID = []
EncodedPixels = []
num = 0
conv = lambda l: ' '.join(map(str, l))

for name in images_name:
    image = cv2.imread(inpath + name)
    outputs = predictor(image)
    print(num)
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

## def cat for inf and show
dataset_dicts = DatasetCatalog.get('multiclass_Test')
for i, d in enumerate(random.sample(dataset_dicts,5)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=multiclass_test_metadata, scale=0.5)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (15, 20))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    # plt.savefig(f"./Results/Output_{i}.png")
plt.show()



# ## single image inference, now obsolete, but rewrite to standalone function

# image_path = "/content/drive/MyDrive/Colab Notebooks/UW/COM_Vision/DATASETS/D_train/images/01213_A.000006.tif"


# def on_image(image_path, predictor):
#     im = cv2.imread(image_path)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=multiclass_test_metadata,
#                    scale=0.5,
#                    instance_mode=ColorMode.IMAGE_BW)
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

#     plt.figure(figsize=(15, 20))
#     #plt.imshow(v.get_image())
#     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#     plt.show()



# on_image(image_path, predictor)



## def for analysis and measurements
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

## sub inference from mask
def GetInference():
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                  metadata=multiclass_test_metadata,
                  scale=1,
                  instance_mode=ColorMode.SEGMENTATION)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  # cv2.imwrite(out.get_image()[:, :, ::-1])

## count types
def GetCounts():
  outputs = predictor(im)
  classes = outputs["instances"].pred_classes.to("cpu").numpy()
  TotalCount = sum(classes==1)+sum(classes==2)+sum(classes==3)+sum(classes==4)
  ParticleCount = sum(classes==1)
  BubbleCount = sum(classes==2)
  DropletCount = sum(classes==3)
  PList.append(ParticleCount)
  DList.append(DropletCount)
  BList.append(BubbleCount)
  # things_classes=["Scale bar","Wall thickness of polyHIPEs","Pore throats of polyHIPEs","Pores of polyHIPEs"])


## get mask contours for outlines / ferret
def GetMask_Contours():
  outputs = predictor(im)
  mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
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
      Length = min(dimA, dimB)
      Width = max(dimA, dimB)
      CircularED = np.sqrt(4*area/np.pi)
      Chords = cv2.arcLength(c,True)
      Roundness = 1/(Aspect_Ratio) if Aspect_Ratio != 0 else 0
      Sphericity = (2*np.sqrt(np.pi*dimArea))/dimPerimeter
      Circularity = 4*np.pi*(dimArea/(dimPerimeter)**2)
      Feret_diam = diaFeret
      lengthList.append(Length)
      widthList.append(Width)
      circularEDList.append(CircularED)
      aspectRatioList.append(Aspect_Ratio)
      circularityList.append(Circularity)
      chordsList.append(Chords)
      ferretList.append(Feret_diam)
      roundList.append(Roundness)
      sphereList.append(Sphericity)


## create and append lists
lengthList = list()
widthList = list()
circularEDList = list()
aspectRatioList = list()
circularityList = list()
chordsList = list()
ferretList = list()
roundList = list()
sphereList = list()
PList = list()
DList = list()
BList = list()
tPL = 0
tBL = 0
tDL = 0
count = 0

test_img_path = "/home/deamoon_uw_nn/DATASET/INFERENCE/"
x_th = len(test_img_path)
x_c = 0

for test_img in os.listdir(test_img_path):

    input_path = os.path.join(test_img_path, test_img)
    im = cv2.imread(input_path)
    GetInference()
    GetCounts()
    GetMask_Contours()

    count = count+1

# #moving avgs
window_size = 3
i = 0
k = 0

MA_lengthList = []
MA_widthList = []
MA_circularEDList = []
MA_aspectRatioList = []
MA_circularityList = []
MA_chordsList = []
MA_ferretList = []
MA_roundList = []
MA_sphereList = []

lists = [lengthList,widthList,circularEDList,aspectRatioList,circularityList,chordsList,ferretList,roundList,sphereList]
listnames = ['lengthList','widthList','circularEDList','aspectRatioList','circularityList','chordsList','ferretList','roundList','sphereList']

for lst in lists:

  listname_str = 'MA_' + listnames[k]
  k = k+1

  while i < (len(lst) - window_size + 1):
      window = lst[i : i + window_size]
      window_average = round(sum(window) / window_size, 2)
      vars()[listname_str].append(window_average)
      i = i+1

  i = 0

lengthBins = np.histogram(np.asarray(MA_lengthList))
widthBins = np.histogram(np.asarray(MA_widthList))
circularEDBins = np.histogram(np.asarray(MA_circularEDList))
aspectRatioBins = np.histogram(np.asarray(MA_aspectRatioList))
circularityBins = np.histogram(np.asarray(MA_circularityList))
chordsBins = np.histogram(np.asarray(MA_chordsList))
ferretBins = np.histogram(np.asarray(MA_ferretList))
roundBins = np.histogram(np.asarray(MA_roundList))
sphereBins = np.histogram(np.asarray(MA_sphereList))

for PL in range(0, len(PList)):
    tPL = tPL + PList[PL]
for DL in range(0, len(DList)):
    tDL = tDL + DList[DL]
for BL in range(0, len(BList)):
    tBL = tBL + BList[BL]

values = list()
values.append(tPL)
values.append(tBL)
values.append(tDL)
values = [*values, *lengthBins, *widthBins, *circularEDBins, *circularityBins, *chordsBins]
print("No. (AVG) of Particles, Bubbles, Droplets:  " + repr(tPL/count) + ",  "+ repr(tBL/count)+ ",  "+ repr(tDL/count))
print("No. (Total) of Particles, Bubbles, Droplets:  " + repr(tPL) + ",  "+ repr(tBL)+ ",  "+ repr(tDL))
print("No. of images / no. of images used:  " + repr(x_c) + "  /  "+ repr(count))

rows = zip(MA_ferretList,MA_aspectRatioList,MA_roundList,MA_circularityList,MA_sphereList,MA_lengthList,MA_widthList,MA_circularEDList,MA_chordsList)

with open('ShapeDescriptor.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

df = pd.read_csv('ShapeDescriptor.csv', header=None)
df.columns = ['Feret Diameter', 'Aspect Ratio', 'Roundness', 'Circularity', 'Sphericity', 'Length', 'Width', 'CircularED', 'Chords']
df.to_csv('Results.csv', index=True)

sns.displot(df['Feret Diameter'])
sns.displot(df['Aspect Ratio'])
sns.displot(df['Roundness'])
sns.displot(df['Circularity'])
sns.displot(df['Sphericity'])
sns.displot(df['CircularED'])
sns.displot(df['Chords'])
sns.displot(df['Feret Diameter'], kind='kde')
sns.displot(df['Aspect Ratio'], kind='kde')
sns.displot(df['Roundness'], kind='kde')
sns.displot(df['Circularity'], kind='kde')
sns.displot(df['Sphericity'], kind='kde')
sns.displot(df['CircularED'], kind='kde')
sns.displot(df['Chords'], kind='kde')
