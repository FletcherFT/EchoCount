from mrcnn.config import Config
from mrcnn import utils
from skimage import io, draw, color
import numpy as np
import json
from pathlib import Path
from shapely.geometry import LineString
import imagesize


class EchoConfig(Config):
    """Configuration class for training on the echogram dataset.
    Derives from the base Config class and overrides values specific
    to the echogram dataset.
    """
    # Give the configuration a recognizable name
    NAME = "echogram"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1  # GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # IMAGES_PER_GPU = 8

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + SingleTarget class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 1

    # Image mean (RGB)
    MEAN_PIXEL = np.array([128])

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 24, 32, 48)  # anchor side in pixels

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (128, 128)  # (height, width) of the mini-mask

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 300

    ROI_POSITIVE_RATIO = 0.33

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1024

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 256

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.9

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    TRAIN_BN = False


class EchoDataset(utils.Dataset):
    """Generates the echogram dataset. The dataset consists of echograms containing cod instances.
    """

    def _polyline_to_polygon(self, polyline, thickness=3):
        """Convert a polyline annotation into a polygon annotation by thickening using Shapely library"""
        assert polyline["shape_attributes"]["name"] == "polyline"
        x = polyline["shape_attributes"]["all_points_x"]
        y = polyline["shape_attributes"]["all_points_y"]
        xy = list(zip(x, y))
        line = LineString(xy)
        p = list(line.buffer(thickness).exterior.coords)
        x, y = zip(*p)
        polyline["shape_attributes"]["all_points_x"] = x
        polyline["shape_attributes"]["all_points_y"] = y
        polyline["shape_attributes"]["name"] = "polygon"
        return polyline

    def _unpack_via2_json_annotation(self, jpath, thickness=3):
        agg_annotations = []
        for jfile in jpath:
            with open(jfile) as f:
                annotations = json.load(f)
                for annotation in annotations.values():
                    for i, region in enumerate(annotation["regions"]):
                        if region["shape_attributes"]["name"] == "polyline":
                            annotation["regions"][i] = self._polyline_to_polygon(region)
                    agg_annotations.append(annotation)
        return agg_annotations

    def _unpack_via2_json_project(self, jpath, thickness=3):
        agg_annotations = []
        for jfile in jpath:
            with open(jfile) as f:
                annotations = json.load(f)["_via_img_metadata"]
                for annotation in annotations.values():
                    for i, region in enumerate(annotation["regions"]):
                        if region["shape_attributes"]["name"] == "polyline":
                            annotation["regions"][i] = self._polyline_to_polygon(region)
                    agg_annotations.append(annotation)
        return agg_annotations

    def load_echogram(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        assert subset in ["train", "valid", "tests"], "No subset called {}".format(subset)
        # Add classes
        self.add_class("echogram", 1, "singletarget")
        dataset_path = Path(dataset_dir).resolve()
        json_files = list(dataset_path.glob("{}.json".format(subset)))
        annotations = self._unpack_via2_json_annotation(json_files)

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stored in the
            # shape_attributes.
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = Path(dataset_dir, a['filename']).resolve()
            width, height = imagesize.get(image_path)

            self.add_image(
                "echogram",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def image_reference(self, image_id):
        """Return the echogram data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "echogram":
            return info["echogram"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not an echogram dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "echogram":
            return super(self.__class__, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'], shape=mask.shape[:2])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
                """
        # Load image
        image = io.imread(self.image_info[image_id]['path'])
        # If has an alpha channel, remove it for consistency
        if image.ndim > 3:
            image = image[..., :3]
        if image.ndim > 2:
            image = color.rgb2gray(image)
            image = np.expand_dims(image, axis=-1)
        return image
