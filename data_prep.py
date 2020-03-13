from sklearn import model_selection
import numpy as np
import glob
from via_tools.via_tools import via32dict


def train_val_test_split_images(images, train_split=-1.0, val_split=-1.0, test_split=-1.0):
    """train_val_test_split_images This function takes in a list of paths to image files and splits them randomly
    into training, validation, and test data sets. Make sure to specify at least two of the splits (must not sum
    above 1.0!)
    Inputs:
    images: list of path names to images.
    train_split: training ratio, default is -1.0
    val_split: validation ratio, default is -1.0
    test_split: test ratio, default it -1.0
    Outputs:
    train: list containing images in training set.
    val: list containing images in validation set.
    test: list containing images in test set."""
    splits = np.array([train_split, val_split, test_split])
    idx = splits < 0
    # Check that there are two split ratios.
    assert idx.sum() <= 1, "At least two split ratios must be given!"
    assert splits[idx].sum() <= 1.0, "Given split ratios sum greater than 1.0!"
    if idx.any():
        splits[idx] = 1.0-splits[np.bitwise_not(idx)].sum()
    # split the training set from the others
    train_test = model_selection.train_test_split(images, train_size=splits[0], test_size=splits[2], shuffle=True)
    train = train_test[0]
    test = train_test[1]
    val = list(set(images) - set(train+test))
    return train, val, test


def split_via3_csv(via3_csv, train, val, test):
    via3_dict = via32dict(via3_csv)
    return split_via3_dict(via3_dict, train, val, test)


def split_via3_dict(via3_dict, train, val, test):
    train_dict = via3_dict(train)
    val_dict = via3_dict(val)
    test_dict = via3_dict(test)
    return train_dict, val_dict, test_dict

images = glob.glob("C:\\Users\\fletho\\Desktop\\Greenland2019ImageExtractions\\data_sets\\*.png")
train, val, test = train_val_test_split_images(images, train_split=0.7, test_split=0.2)
via3_csv = "C:\\Users\\fletho\\Desktop\\Greenland2019ImageExtractions\\data_sets\\test_labels.csv"
train_dict, val_dict, test_dict = split_via3_csv(via3_csv, train, val, test)
