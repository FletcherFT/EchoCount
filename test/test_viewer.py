from skimage import io, viewer, util, draw
from pathlib import Path
import argparse
import json
import numpy as np
import copy


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", nargs="?", default="Data", type=str,
                        help="Directory pointing to train, valid, tests subsets.")
    parser.add_argument("subset", nargs="?", default="all", type=str, choices=["train", "valid", "tests", "all"],
                        help="Data subset to inspect. Default is to include all subsets.")
    parser.add_argument("--window_size", nargs="?", default=0, type=int,
                        help="Size of the sliding window to view the images over, 0 indicates full size.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    if args.subset == "all":
        data_path = data_dir.joinpath("Annotations.json")
    else:
        data_path = data_dir.joinpath("{}.json".format(args.subset))
    with data_path.open(mode="r") as f:
        annotations = json.load(f)
    raw_collection = []
    img_collection = []
    for annotation in annotations.values():
        if not args.window_size:
            raw_collection.append(io.imread(data_dir.joinpath(annotation["filename"])))
        else:
            img = io.imread(data_dir.joinpath(annotation["filename"]))
            h, w, c = img.shape
            r_pad = args.window_size - (h % args.window_size)
            c_pad = args.window_size - (w % args.window_size)
            img = util.pad(img, ((0, r_pad), (0, c_pad), (0, 0)))
            windows = util.view_as_blocks(img, (args.window_size, args.window_size, 3)).squeeze()
            m, n, _, _, _ = windows.shape
            for r in range(m):
                for c in range(n):
                    r_min = r * args.window_size
                    r_max = (r + 1) * args.window_size - 1
                    c_min = c * args.window_size
                    c_max = (c + 1) * args.window_size - 1
                    raw_collection.append(windows[r, c, :, :, :])
                    roi = windows[r, c, :, :, :].copy()
                    regions = copy.deepcopy(annotation["regions"])
                    for region in regions:
                        # Get the points of a region line (zero indexed)
                        xy = np.array([region["shape_attributes"]["all_points_x"],
                                       region["shape_attributes"]["all_points_y"]]) - 1
                        flags = np.all(np.bitwise_and(xy <= np.array([[c_max], [r_max]]),
                                                    xy >= np.array([[c_min], [r_min]])), axis=0)
                        # Check that the region is within the ROI window
                        if np.any(flags):
                            # Choose to keep rising and falling edges
                            keep_ind = []
                            for i, val in enumerate(flags):
                                if i == 0:
                                    prev = val
                                    continue
                                # Rising Edge
                                if val and not prev:
                                    keep_ind.append(i - 1)
                                # Falling Edge
                                if not val and prev:
                                    keep_ind.append(i)
                                # Region inside
                                if val:
                                    keep_ind.append(i)
                                # Update previous
                                prev = val
                            # Keep the components of the region that are in ROI.
                            xy_clipped = xy[:, keep_ind]
                            #xy_clipped = xy[:, flags]
                            # Clip the region to the boundaries of the ROI window
                            xy_clipped = np.clip(xy_clipped, np.array([[c_min], [r_min]]), np.array([[c_max], [r_max]]))
                            # Offset the region to the ROI window coordinates
                            xy_clipped = xy_clipped - np.array([[c_min], [r_min]])
                            # Draw the line between each point
                            for j in range(xy_clipped.shape[1] - 1):
                                cc, rr = draw.line(xy_clipped[0, j], xy_clipped[1, j],
                                                   xy_clipped[0, j+1], xy_clipped[1, j+1])
                                roi[rr, cc, 0] = 255
                    img_collection.append(np.hstack((windows[r, c, :, :, :], roi)))
    view = viewer.CollectionViewer(img_collection)
    view.show()
