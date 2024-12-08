import json
import os
from typing import Dict


def horizontal_flip(general: Dict) -> Dict:
    for data_dict in general["data"]:
        width = data_dict["img_width"]
        for i, (xmin, ymin, xmax, ymax) in enumerate(data_dict["gt_boxes"]):
            data_dict["gt_boxes"][i] = [width - xmax, ymin, width - xmin, ymax]
        for i, (xmin, ymin, xmax, ymax) in enumerate(data_dict["pd_boxes"]):
            data_dict["pd_boxes"][i] = [width - xmax, ymin, width - xmin, ymax]
    return general


def horizontal_flip_io(ant_path: str, save_path: str):
    general = json.load(open(ant_path, "r"))
    general = horizontal_flip(general)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(general, f, indent=4)
