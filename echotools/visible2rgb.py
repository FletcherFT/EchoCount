import numpy as np
import cv2
from skimage import data, color, io
import argparse
import os


def frequency_shift_visible(spectra, sp_min=None, sp_max=None, celerity=299792458):
    """Shift frequency spectra into visible spectrum.
    Inputs
    spectra: iterable containing frequencies to be mapped to visible spectrum (Hz)
    sp_min (optional): value for minimum frequency (Hz), defaults to minimum of spectra.
    sp_max (optional): value for maximum frequency (Hz), defaults to maximum of spectra.
    celerity (optional): value for speed of medium (m/s), defaults to light speed in vacuum."""
    if sp_min is None:
        sp_min = min(spectra)
    if sp_max is None:
        sp_max = max(spectra)

    # convert spectra into wavelength
    wavelength = spectra * celerity
    wavelength_min = sp_min * celerity
    wavelength_max = sp_max * celerity

    # interpolate spectra wavelength into visible light wavelength
    return np.interp(wavelength, [wavelength_min, wavelength_max], [380e-9, 750e-9])


def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    # scale the wavelength to nm
    wavelength = wavelength * 1e9
    # initialize the RGB output
    RGB = np.zeros([3, wavelength.shape[0]])
    # Find the wavelength in the blue/violet spectra
    idx = np.bitwise_and(np.less_equal(380, wavelength), np.less_equal(wavelength, 440))
    if np.any(idx):
        attenuation = 0.3 + 0.7 * (wavelength[idx] - 380) / (440 - 380)
        RGB[:, idx] = np.array([((-(wavelength[idx] - 440) / (440 - 380)) * attenuation) ** gamma,
                                [np.zeros((1, wavelength[idx].size))],
                                (1 * attenuation) ** gamma])
    idx = np.bitwise_and(np.less(440, wavelength), np.less_equal(wavelength, 490))
    if np.any(idx):
        RGB[:, idx] = np.array([[np.zeros((1, wavelength[idx].size))],
                                ((wavelength[idx] - 440) / (490 - 440)) ** gamma,
                                [np.ones((1, wavelength[idx].size))]])
    idx = np.bitwise_and(np.less(490, wavelength), np.less_equal(wavelength, 510))
    if np.any(idx):
        RGB[:, idx] = np.array([[np.zeros((1, wavelength[idx].size))],
                                [np.ones((1, wavelength[idx].size))],
                                (-(wavelength[idx] - 510) / (510 - 490)) ** gamma])
    idx = np.bitwise_and(np.less(510, wavelength), np.less_equal(wavelength, 580))
    if np.any(idx):
        RGB[:, idx] = np.array([((wavelength[idx] - 510) / (580 - 510)) ** gamma,
                                [np.ones((1, wavelength[idx].size))],
                                [np.zeros((1, wavelength[idx].size))]])
    idx = np.bitwise_and(np.less(580, wavelength), np.less_equal(wavelength, 645))
    if np.any(idx):
        RGB[:, idx] = np.array([[np.ones((1, wavelength[idx].size))],
                                (-(wavelength - 645) / (645 - 580)) ** gamma,
                                [np.zeros((1, wavelength[idx].size))]])
    idx = np.bitwise_and(np.less(645, wavelength), np.less_equal(wavelength, 750))
    if np.any(idx):
        attenuation = 0.3 + 0.7 * (750 - wavelength[idx]) / (750 - 645)
        RGB[:, idx] = np.array([(1 * attenuation) ** gamma,
                                [np.zeros((1, wavelength[idx].size))],
                                [np.zeros((1, wavelength[idx].size))]])
    idx = np.bitwise_or(np.less(wavelength, 380), np.less(750, wavelength))
    if np.any(idx):
        RGB[:, idx] = np.zeros((3, wavelength[idx].size))
    RGB = RGB * 255.0
    return RGB.astype('uint8')


def generate_composite(alphas, layers):
    """Create composite image by encoding image values as an alpha channel, and overlaying
    each layer as a separate color.
    """
    # First get the blending alphas for each image.
    for i, alpha in enumerate(alphas):
        if len(alpha.shape) > 2:
            alpha = color.rgb2gray(alpha)
        if alpha.dtype is np.dtype('uint8'):
            # if uint8, normalize the image
            alpha = alpha.astype('float64')
            alpha = alpha / 255.0
        if i == 0:
            weights = np.zeros(list(alpha.shape) + [len(alphas)])
        weights[:, :, i] = alpha
    # Because alpha is normalized, the summation of weights can be maximum of the number of alphas, so normalize.
    weights = weights / len(alphas)
    height, width = weights.shape[0:2]

    # Initialise the composite image
    composite = np.zeros((height, width, 3))
    for i in range(len(alphas)):
        layer = layers[:, i].astype('float64')
        layer = layer / 255.0
        image = np.zeros((height, width, 3))
        for j in range(3):
            image[:, :, j] = layer[j]
        # Add the weighted image to the composite
        weight = np.concatenate([np.expand_dims(weights[:, :, i], -1) * 3], axis=2)
        composite = composite + image * weight
    return (composite * 255.0).astype('uint8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift arbitrary echograms into spectral region, and compose the colour")
    parser.add_argument("images", nargs="+", help="Set of echograms to use.")
    parser.add_argument("-s", "--spectra", type=float, nargs="+", help="Corresponding frequency for each echogram (Hz).", required=True)
    parser.add_argument("-c", "--celerity", nargs="?", help="Specify celerity of input medium (m/s).", type=float, default=299792458)
    parser.add_argument("-r", "--range", nargs=2, help="Specify lower and upper range (Hz)", type=float)
    parser.add_argument("--invert", action="store_true", help="Flag whether to invert the input images or not.")
    parser.add_argument("--save", action="store_true", help="Flag whether to save the composite image or not.")
    args = parser.parse_args()
    spectra = np.array(args.spectra)
    if args.range is None:
        specmin = spectra.min()
        specmax = spectra.max()
    else:
        specmin = min(args.range)
        specmax = max(args.range)
    visible = frequency_shift_visible(spectra, specmin, specmax, args.celerity)
    RGB = wavelength_to_rgb(visible)
    imgs = [io.imread(os.path.abspath(i)) for i in args.images]
    if args.invert:
        imgs = [255-img for img in imgs]
    composite = generate_composite(imgs, RGB)
    if args.save:
        parent, basename = os.path.split(os.path.abspath(args.images[0]))
        if args.invert:
            io.imsave(os.path.join(parent, "inverted_spectral_composition_{}".format(basename)), composite)
        else:
            io.imsave(os.path.join(parent, "spectral_composition_{}".format(basename)), composite)
