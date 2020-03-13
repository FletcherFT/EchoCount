import os
from pathlib import Path
from skimage import util, io, color
import argparse


def tile(img_dirs):
    """"""
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]
    for img_dir in args.dir:
        # get image directory
        img_dir = Path(img_dir)
        if not img_dir.is_absolute():
            img_dir = img_dir.absolute()
        # get output directory
        output_dir = img_dir.joinpath("tiles")
        output_dir.mkdir(parents=True, exist_ok=True)
        # get images within image directory
        images = []
        extensions = ('*.tif', '*.png', '*.jpg', '*.jpeg')
        for ext in extensions:
            images.extend(img_dir.glob(ext))
        for image in images:
            basename, extension = os.path.splitext(image)
            img = io.imread(image)
            if len(img.shape) == 2:
                img = color.gray2rgb(img)
            # METHOD 1: SLICE IMAGE INTO WXW SQUARES
            height, width, channels = img.shape
            pad_height = height % width
            pad_width = width % width
            img = util.pad(img, ((0, width - pad_height), (0, 0), (0, 0)))
            height, width, channels = img.shape
            GRID_HEIGHT = int(height / width)
            GRID_WIDTH = int(width / width)
            tiles = img.reshape((GRID_HEIGHT, int(height / GRID_HEIGHT),
                                 GRID_WIDTH, int(width / GRID_WIDTH), img.shape[2])).swapaxes(1, 2)
            m, n, height, width, channels = tiles.shape
            for i in range(m):
                for j in range(n):
                    # convert image into png format (VIA doesn't like tiff)
                    out_name = os.path.join(output_dir, "{}_{}-{}.png".format(os.path.split(basename)[-1], i, j))
                    tile = tiles[i, j, :, :, :]
                    io.imsave(out_name, tile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a file listing file URIs of a directory containing images.')
    parser.add_argument('-d', '--dir', metavar='in_dir', type=str, nargs='+', default=['.'],
                        help='directory containing the images (global or relative to directory this script was invoked)')
    args = parser.parse_args()
    tile(args.dir)
