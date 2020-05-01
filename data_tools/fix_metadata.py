from pathlib import Path
import json


if __name__ == "__main__":
    # Get the data directory of interest
    data_dir = Path("../raw_data/LabelFiles").resolve()
    # Get the annotation file in the data directory
    data_paths = list(data_dir.glob("*.json"))
    # Load in the json as a dictionary
    annotation_output = dict()
    project_output = dict()
    for i, data_path in enumerate(data_paths):
        with open(data_path) as f:
            project = json.load(f)
        project["_via_attributes"]["file"] = {"metadata": {"type": "checkbox",
                                                           "description": "Contains metadata for annotated files.",
                                                           "options": {"isnegative": "if true then negative image",
                                                                       "isfinished": "if true then annotation done"},
                                                           "default_options": {}}}
        for annotation in project["_via_img_metadata"].keys():
            metadata = {"isnegative": False, "isfinished": False}
            if "Negative" in project["_via_img_metadata"][annotation]["file_attributes"] and project["_via_img_metadata"][annotation]["file_attributes"]["Negative"]:
                metadata["isnegative"] = project["_via_img_metadata"][annotation]["file_attributes"]["Negative"]["isnegative"]
            else:
                metadata["isnegative"] = False
            if "Finished" in project["_via_img_metadata"][annotation]["file_attributes"] and project["_via_img_metadata"][annotation]["file_attributes"]["Finished"]:
                metadata["isfinished"] = project["_via_img_metadata"][annotation]["file_attributes"]["Finished"]["isfinished"]
            else:
                metadata["isfinished"] = False
            project["_via_img_metadata"][annotation]["file_attributes"] = {"metadata": metadata}
            annotation_output[annotation] = project["_via_img_metadata"][annotation]
    project_output["_via_settings"] = project["_via_settings"]
    project_output["_via_img_metadata"] = annotation_output
    project_output["_via_attributes"] = project["_via_attributes"]
    with open(data_dir.joinpath("CombinedProject.json".format(i)), "w") as f:
        json.dump(project_output, f)
    with open(data_dir.joinpath("Annotations.json".format(i)), "w") as f:
        json.dump(annotation_output, f)