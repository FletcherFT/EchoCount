from pathlib import Path
import json
from skimage import io, draw, viewer
from shapely.geometry import LineString
import matplotlib.pyplot as plt


def _polyline_to_polygon(polyline, thickness=3):
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


if __name__ == "__main__":
    # Get the data directory of interest
    data_dir = Path("../raw_data/Data/Batch_0001").resolve()
    # Get the annotation file in the data directory
    data_path = list(data_dir.glob("Annotations.json"))[0]
    # Load in the json as a dictionary
    with open(data_path) as f:
        annotations = json.load(f)
    imgs = []
    target_number = []
    isdone = [annotation_key for annotation_key in annotations.keys() if annotations[annotation_key]]
    for annotation_key in annotations.keys():
        annotation = annotations[annotation_key]
        isdone.append(annotation["file_attributes"]["metadata"]["isfinished"])
        img = io.imread(data_dir.joinpath(annotation["filename"]))
        target_number.append(len(annotation["regions"]))
        for polyline in annotation["regions"]:
            r = []
            c = []
            for i in range(len(polyline["shape_attributes"]["all_points_y"])-1):
                rr, cc = draw.line(polyline["shape_attributes"]["all_points_y"][i]-1,
                                   polyline["shape_attributes"]["all_points_x"][i]-1,
                                   polyline["shape_attributes"]["all_points_y"][i+1]-1,
                                   polyline["shape_attributes"]["all_points_x"][i+1]-1)
                r+=rr.tolist()
                c+=cc.tolist()
            img[r, c, 0] = 255
            polygon = _polyline_to_polygon(polyline, thickness=3)
            rr, cc = draw.polygon(polygon["shape_attributes"]['all_points_y'], polygon["shape_attributes"]['all_points_x'], shape=img.shape[:2])
            img[rr, cc, 1] = 255
        imgs.append(img)
    negatives = [i for i in target_number if i < 1]
    positives = [i for i in target_number if i > 0]
    f2, ax2 = plt.subplots()
    plt.hist(positives, bins=100)
    ax2.set_xlabel("Number of targets")
    ax2.set_ylabel("Count")
    f2.suptitle("Negative Samples: {}, Positive Samples: {}".format(len(negatives), len(positives)))
    f2.canvas.draw()
    f2.canvas.flush_events()
    V = viewer.CollectionViewer(imgs)
    V.show()
