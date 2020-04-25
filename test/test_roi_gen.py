from pathlib import Path
import json
from skimage import io, draw, viewer
import numpy as np
from threading import Thread
from queue import Queue, Empty
import os
import copy


class Worker(Thread):
    def __init__(self, queue, name=None):
        self._queue = queue
        self._is_stop = False
        super().__init__(target=self.worker, name=name)

    def worker(self):
        while not self._is_stop:
            try:
                fname, img = self._queue.get_nowait()
                print("Saving {}".format(str(fname)))
                io.imsave(fname, img)
                self._queue.task_done()
            except Empty:
                pass

    def stop(self):
        self._is_stop = True


if __name__ == "__main__":
    n_workers = 100
    image_queue = Queue()
    workers = [Worker(image_queue, "Worker-{:02d}".format(i)) for i in range(n_workers)]
    for w in workers:
        w.start()
    # Get the data directory of interest
    data_dir = Path("../raw_data/Data/Batch_0001").resolve()
    # Get the annotation file in the data directory
    data_path = list(data_dir.glob("Annotations.json"))[0]
    # Load in the json as a dictionary
    with open(data_path) as f:
        annotations = json.load(f)
    output_dir = data_dir.joinpath("Processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    window = (256, 256)
    rois = []
    x_jump = 1 # jump in column
    y_jump = 1 # jump in row
    # create a new dictionary for the updated annotations file
    new_annotations = dict()
    # Go through each annotation
    for annotation in annotations.values():
        # Check if the annotation is complete
        if annotation['file_attributes']["metadata"]["isfinished"]:
            # Get the filename
            fname = data_dir.joinpath(annotation["filename"])
            # Get the regions in the current annotation
            regions = annotation["regions"]
            # read in the image
            img = io.imread(fname)
            # get the image dimensions
            width, height, channels = img.shape
            # init the row and column indices
            r = 0
            c = 0
            while r < height-window[0]:
                while c < width-window[1]:
                    # access the region of interest
                    roi = img[r:r+window[0], c:c+window[1], :]
                    # append the list of rois
                    #rois.append(roi)
                    # Save the image
                    stem = fname.stem
                    out_name = output_dir.joinpath("{}_{}-{}.png".format(stem, r, c))
                    image_queue.put((out_name, roi))
                    # Initialise a list of flags to indicate if region is inside (border included) or outside of the roi
                    flags = []
                    # Initialise the roi regions
                    roi_regions = copy.deepcopy(regions)
                    # Draw a plot
                    #fig, ax = plt.subplots()
                    # Copy the roi array for anotation
                    #roi_annotated = roi.copy()
                    # Go through each of the regions
                    for i, region in enumerate(regions):
                        # Get the points of a region line (zero indexed)
                        xy = np.array([region["shape_attributes"]["all_points_x"],
                                       region["shape_attributes"]["all_points_y"]])-1
                        # Indicate if region outside (true) or inside (border included)
                        flags.append(np.all(np.any(np.bitwise_or(xy <= np.array([[c], [r]]),
                                                                 xy >= np.array([[c + window[1]], [r + window[0]]])),
                                                   axis=0)))
                        # Clip the line to the boundary
                        xy = np.clip(xy, np.array([[c], [r]]), np.array([[c+window[1]-1], [r+window[0]-1]]))
                        xy = xy - np.array([[c], [r]])
                        # Store the clipped line into the roi_region
                        roi_regions[i]["shape_attributes"]["all_points_x"] = (xy[0,:]+1).tolist()
                        roi_regions[i]["shape_attributes"]["all_points_y"] = (xy[1,:]+1).tolist()
                        #if not flags[-1]:
                        #    for j in range(xy.shape[1]-1):
                        #        cc, rr = draw.line(xy[0, j], xy[1, j], xy[0, j+1], xy[1, j+1])
                        #        roi_annotated[rr, cc, 0] = 255
                    #io.imshow(roi_annotated)
                    #fig.canvas.draw()
                    # Remove the regions that were flagged as outside of the roi
                    roi_regions = [region for i, region in enumerate(roi_regions) if not flags[i]]
                    # Update the annotation's regions property
                    base_name = out_name.name
                    new_annotations[base_name] = {"filename": base_name,
                                                  "size": 0,
                                                  "regions": roi_regions,
                                                  "file_attributes": annotation["file_attributes"]}
                    # increment the column index
                    c += x_jump
                # increment the row index
                r += y_jump
                c = 0
    # Wait here until all images are processed
    image_queue.join()
    # stop the workers
    for w in workers:
        w.stop()
        w.join()
    # Get the sizes of all the images
    output_images = output_dir.glob("*.png")
    # Do a final update of the annotations to complete the information
    for img in output_images:
        base_name = img.name
        size = os.stat(img).st_size
        new_annotations[base_name]["size"] = size
        new_annotations["{}{}".format(base_name, size)] = new_annotations.pop(base_name)
    # output the annotations into the new folder
    fname = output_dir.joinpath("Annotations.json")
    with open(fname, 'w') as f:
        json.dump(new_annotations, f)
    # get the project file
    project_path = list(data_dir.glob("CombinedProject.json"))[0]
    with open(project_path) as f:
        project = json.load(f)
    new_project = project
    new_project["_via_img_metadata"] = new_annotations
    fname = output_dir.joinpath("CombinedProject.json")
    with open(fname, 'w') as f:
        json.dump(new_project, f)
    #V = viewer.CollectionViewer(rois)
    #V.show()
