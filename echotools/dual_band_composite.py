import os
from pathlib import Path
from skimage import util, io, color
import argparse
import difflib
import numpy as np


def compose(img_dirs, bandwidths, save=False, invert=False):
    """Return and optionally save a composition image of echogram bandwidths. Each echogram is encoded into a
    separate channel, meaning there can only be three echogram bandwidths allowed.
    Inputs
    img_dirs: a string containing a directory, or a list of directories where echogram groups are contained.
    bandwidths: a length 3 list of frequency bands, the bands will be encoded into R, G, B respectively, an empty string
    means that the color channel is set to zero.
    save: a boolean indicating whether the composite image should be saved. If true, images are saved in the respective
    source directory prepended with 'composite'
    invert: choose whether to invert the grayscale of the incoming images"""
    assert len(bandwidths) == 3, "Length of bandwidths must be 3."
    L = [len(i) > 0 for i in bandwidths]
    assert sum(L) >= 1, "At least one channel of bandwidths must not be empty."
    if img_dirs is str:
        img_dirs = [img_dirs]
    for img_dir in img_dirs:
        # get image directory
        img_dir = Path(img_dir)
        if not img_dir.is_absolute():
            img_dir = img_dir.absolute()
        # get output directory
        output_dir = img_dir
        # get images within image directory
        images = []
        extensions = ('*.tif', '*.png', '*.jpg', '*.jpeg')
        for ext in extensions:
            images.extend(img_dir.glob(ext))

        echopath = {"R": [], "G": [], "B": []}
        keys = list(echopath.keys())

        for i in range(3):
            if L[i]:
                echopath[keys[i]] = [str(image) for image in images if os.path.split(image)[-1].startswith(bandwidths[i])]
                I = len(echopath[keys[i]])

        for i in range(3):
            if not L[i]:
                echopath[keys[i]] = [""]*I
        if L[0] and L[1]:
            mappingRG = [difflib.get_close_matches(R, echopath["G"], n=1)[0] for R in echopath["R"]]
            echopath["G"] = [echopath["G"][i] for i in mappingRG]
        if L[0] and L[2]:
            mappingRB = [difflib.get_close_matches(R, echopath["B"], n=1)[0] for R in echopath["R"]]
            echopath["B"] = [i for i in mappingRB]
        if L[1] and L[2]:
            mappingGB = [difflib.get_close_matches(G, echopath["B"], n=1)[0] for G in echopath["G"]]
            echopath["B"] = [i for i in mappingGB]

        composites = []
        for R, G, B in zip(echopath["R"], echopath["G"], echopath["B"]):
            if len(R) > 0:
                imR = io.imread(R, as_gray=True)
                if invert:
                    imR = 1.0-imR
                height, width = imR.shape[:2]
            if len(G) > 0:
                imG = io.imread(G, as_gray=True)
                if invert:
                    imG = 1.0-imG
                height, width = imG.shape[:2]
            if len(B) > 0:
                imB = io.imread(B, as_gray=True)
                if invert:
                    imB = 1.0-imB
                height, width = imB.shape[:2]
            composite = np.zeros((height, width, 3))
            if L[0]:
                composite[:, :, 0] = imR
                path = R
                prefix = bandwidths[0]
            if L[1]:
                composite[:, :, 1] = imG
                path = G
                prefix = bandwidths[1]
            if L[2]:
                composite[:, :, 2] = imB
                path = B
                prefix = bandwidths[2]
            composite = composite * 255.0
            composite = composite.astype('uint8')
            composites.append(composite)
            if save:
                _, basename = os.path.split(path)
                basename = basename.split(prefix)[1].strip()
                if invert:
                    out_name = os.path.join(output_dir, "inverted_composite_{}".format(basename))
                    for i in range(3):
                        if L[i]:
                            out_name = os.path.join(output_dir, "inverted_{} {}".format(bandwidths[i], basename))
                            io.imsave(out_name, composite[:, :, i])
                else:
                    out_name = os.path.join(output_dir, "composite_{}".format(basename))
                    io.imsave(out_name, composite)

        return composites


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a composite image out of up to three echogram bandwidths from a '
                    'directory.')
    parser.add_argument('-b', '--bw', type=str, nargs='+', default=['38', '', '200'],
                        help='directory containing the images (global or relative to directory this script was invoked)')
    parser.add_argument('-d', '--dir', type=str, nargs='+', default=['.'],
                        help='directory containing the images (global or relative to directory this script was invoked)')
    parser.add_argument('-s', '--save', action="store_true", default=False,
                        help='directory containing the images (global or relative to directory this script was invoked)')
    parser.add_argument('-i', '--invert', action="store_true", default=False,
                        help='directory containing the images (global or relative to directory this script was invoked)')
    args = parser.parse_args()
    compose(args.dir, args.bw, args.save, args.invert)
