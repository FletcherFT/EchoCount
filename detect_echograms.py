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


import mrcnn.model as modellib
from mrcnn import utils
from echogram import EchoConfig, EchoDataset
import argparse
from pathlib import Path
from skimage import viewer, util, draw, segmentation, io
import numpy as np


def mask2poly(label_images):
    label_images = np.transpose(label_images, [2, 0, 1])
    for i, label_image in enumerate(label_images):
        label_images[i, :, :] = segmentation.find_boundaries(label_image, connectivity=1, mode='inner', background=0)
    label_images = np.transpose(label_images, [1, 2, 0])
    return label_images


def load(data_obj, window_size):
    # First, get the images
    input_collection = []
    output_collection = []
    mask_collection = []
    for image_id in data_obj.image_ids:
        image_in = data_obj.load_image(image_id)
        mask_out, classes = data_obj.load_mask(image_id)
        h, w, c = image_in.shape
        r_pad = window_size - (h % window_size)
        c_pad = window_size - (w % window_size)
        padding = ((0, r_pad), (0, c_pad), (0, 0))
        image_in = util.pad(image_in, padding)
        mask_out = util.pad(mask_out, padding)
        windows = util.view_as_blocks(image_in, (window_size, window_size, 3)).squeeze()
        m, n, _, _, _ = windows.shape
        for r in range(m):
            for c in range(n):
                crop = (r * window_size, c * window_size, window_size, window_size)
                masks_out = utils.resize_mask(mask_out, 1, padding, crop)
                input_collection.append(windows[r, c, :].copy())
                if masks_out.size > 0:
                    # Extract the boundaries
                    masks_out = mask2poly(masks_out)
                    masks_out = np.any(masks_out, axis=-1)
                    roi = windows[r, c, :].copy()
                    roi[masks_out] = [255, 255, 0]
                    output_collection.append(roi)
                    mask_collection.append(masks_out)
                else:
                    output_collection.append(windows[r, c, :].copy())
                    mask_collection.append(np.zeros(masks_out.shape[:2], dtype=np.bool))
    return input_collection, output_collection, mask_collection


def display(image_collection):
    view = viewer.CollectionViewer(image_collection)
    view.show()


def inspect(data_obj, window_size):
    collections = load(data_obj, window_size)
    display(collections[1])
    return collections


def detect(model_obj, image_collection):
    results_agg = []
    for image in image_collection:
        results_agg += model_obj.detect([image])
    return results_agg


def load_and_detect(data_obj, window_size, model_obj):
    collections = load(data_obj, window_size)
    return detect(model_obj, collections[0]), collections[0], collections[1], collections[2]


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser("Evaluate trained model for MR-RCNN")
    parser.add_argument("data_dir", nargs="?", default="Data", type=str,
                        help="Directory pointing to train, valid, tests subsets.")
    parser.add_argument("subset", nargs="?", default="tests", type=str, choices=["train", "valid", "tests"],
                        help="Data subset to inspect. Default is to include all subsets.")
    parser.add_argument("model", type=str, nargs='?', default='',
                        help="The name of the model you wish to load. Looks in "
                             "the models folder.")
    parser.add_argument("--window_size", nargs="?", default=0, type=int,
                        help="Size of the sliding window to view the images over, 0 indicates full size.")
    args = parser.parse_args()
    # Load dataset
    data_dir = str(Path(args.data_dir).resolve())
    dataset = EchoDataset()
    dataset.load_echogram(data_dir, args.subset)
    dataset.prepare()
    inspect(dataset, args.window_size)
    # Load model
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
    # Configure for Inference
    config = EchoConfig()
    # Modify config for inference
    config.GPU_COUNT = 1
    config.IMAGES_PER_GPU = 1
    config.DETECTION_MIN_CONFIDENCE = 0.7
    # Because the image data is preprocessed, set the resizing mode to None
    config.IMAGE_RESIZE_MODE = "none"
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=COCO_MODEL_DIR)
    model.load_weights(str(COCO_MODEL_PATH), by_name=True)
    # Make detections
    results, inputs, annotated, masks = load_and_detect(dataset, args.window_size, model)
    results_collection = []
    for i, result in enumerate(results):
        mask = mask2poly(result["masks"])
        mask = mask.any(-1)
        img = inputs[i].copy()
        img[mask] = [255, 0, 0]
        ref = annotated[i].copy()
        results_collection.append(np.hstack((ref,
                                             np.array([0, 255, 0])*np.ones((ref.shape[0], 10, 3), dtype=np.int8), img)))
    display(results_collection)
