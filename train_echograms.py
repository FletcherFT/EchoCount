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
import imgaug

parser = argparse.ArgumentParser("Training model for MR-RCNN")
parser.add_argument("data_dir", type=str, nargs='?', default='./data', help="The directory containing the dataset "
                                                                               "(absolute or relative path).")
parser.add_argument("model", type=str, nargs='?', default='', help="The name of the model you wish to load. Looks in "
                                                                   "the models folder.")
parser.add_argument("-e", type=int, nargs='?', default=1, help="Number of epochs to run. (default 1)")
parser.add_argument("-l", type=str, nargs='?', default="all", help="Set trainable layers, can be one of the following:"
                                                                   "all, heads, mask, rpn fpn, 3+, 4+, 5+. Default is"
                                                                   "all.")
args = parser.parse_args()

EPOCHS = args.e
LAYERS = args.l

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
config.display()


def train(model, data_dir, epochs=1, layers="all"):
    """Train the model."""

    data_dir = os.path.abspath(data_dir)
    # Training dataset
    dataset_train = EchoDataset()
    dataset_train.load_echogram(data_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = EchoDataset()
    dataset_val.load_echogram(data_dir, "valid")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers=layers,
                augmentation=imgaug.augmenters.Fliplr(0.5))


# Create Model
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(str(COCO_MODEL_PATH), by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

train(model, args.data_dir, EPOCHS, LAYERS)
