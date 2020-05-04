#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Echogram Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset
# of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though,
# because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to
# get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Echogram* dataset is included below.


import os
import tensorflow as tf
import mrcnn.model as modellib
from mrcnn import utils
from echogram import EchoConfig, EchoDataset
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("Evaluate trained model for MR-RCNN")
parser.add_argument("data_dir", type=str, nargs='?', default='./data', help="The directory containing the dataset "
                                                                            "(absolute or relative path).")
parser.add_argument("model", type=str, nargs='?', default='', help="The name of the model you wish to load. Looks in "
                                                                   "the models folder.")

args = parser.parse_args()

MODEL_DIR = Path("./logs").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
COCO_MODEL_DIR = Path("./models").resolve()
COCO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
if len(args.model) == 0:
    COCO_MODEL_PATH = COCO_MODEL_DIR.joinpath("mask_rcnn_coco.h5")
    if not COCO_MODEL_PATH.exists():
        utils.download_trained_weights(COCO_MODEL_PATH)
else:
    COCO_MODEL_PATH = COCO_MODEL_DIR.joinpath(args.model)
    if not COCO_MODEL_PATH.exists():
        raise Exception("No model file found called: {}".format(COCO_MODEL_PATH))

print(tf.__version__)
config = EchoConfig()
config.GPU_COUNT = 1
config.IMAGES_PER_GPU = 1
config.display()


def evaluate(model, data_dir):
    data_dir = str(Path(data_dir).resolve())
    dataset_test = EchoDataset()
    dataset_test.load_echogram(data_dir, "tests")
    dataset_test.prepare()
    return model.evaluate(dataset_test, steps=10000)


# Create Model
# Create model in inference mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
model.load_weights(str(COCO_MODEL_PATH), by_name=True)
loss = evaluate(model, args.data_dir)
