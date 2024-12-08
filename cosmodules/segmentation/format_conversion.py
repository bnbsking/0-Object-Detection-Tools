from collections import defaultdict
import json
import os
import shutil
from typing import Dict, List, Literal

import cv2
import numpy as np


def get_category_list(categories: List[Dict]):
    max_cat_id = max(category['id'] for category in categories)
    category_list = ["__background__"] * (max_cat_id + 1)
    for cat_dict in categories:
        category_list[cat_dict['id']] = cat_dict['name']
    return category_list


def coco2general(
        img_folder: str,
        ant_path: str,
        save_folder: str,
        contour_width: int = 2
    ):
    """
    The function saves 4 things:
        1. general.json: same format as in detection
        2. *.img: copy the images to the save_folder
        3. gt_contour_*.npy: contour mask in npy format. shape=(h, w) and values are in {0, 1, 2, ..., max_category_id}
        4. gt_filled_*.npy: filled mask in npy format. shape=(categories, h, w) and values are in {0, 1}.
    """
    with open(ant_path, 'r') as f:
        coco = json.load(f)
    os.makedirs(save_folder, exist_ok=True)

    # Get categories and save to json
    general = {
        "categories": get_category_list(coco["categories"]),
        "data": []
    }
    img_id_to_idx = {}

    # update img
    for img_dict in coco["images"]:
        general["data"].append(
            {
                "img_path": os.path.join(save_folder, img_dict["file_name"]),
                "img_width": img_dict["width"],
                "img_height": img_dict["height"],
                "gt_boxes": [],
                "gt_cls": [],
                "gt_contour": [],
                "gt_contour_path": os.path.join(save_folder, "gt_contour_" + img_dict["file_name"].replace(".jpg", ".npy")),
                "gt_filled_path": os.path.join(save_folder, "gt_filled_" + img_dict["file_name"].replace(".jpg", ".npy"))
            }
        )
        img_id_to_idx[img_dict["id"]] = len(general["data"]) - 1

    # update ant
    for ant_dict in coco["annotations"]:
        # change segmentation (x1, y1, ...) into contour [(x1, y1), ...]
        contour = []
        for seg_list in ant_dict["segmentation"]:
            for i in range(len(seg_list) // 2):
                contour.append((seg_list[2 * i], seg_list[2 * i + 1]))
        contour = np.array(contour, dtype=np.int32)
        
        # collect
        idx = img_id_to_idx[ant_dict["image_id"]]
        xmin, ymin, w, h = ant_dict["bbox"]
        general["data"][idx]["gt_boxes"].append(
            [xmin, ymin, xmin + w, ymin + h]
        )
        general["data"][idx]["gt_cls"].append(ant_dict["category_id"])
        general["data"][idx]["gt_contour"].append(contour)

    # Get contour_mask and filled_mask in npy format and save
    for data_dict in general["data"]:
        
        # contour_mask
        contour_mask = np.zeros(
            (data_dict["img_height"], data_dict["img_width"]),
            dtype=np.uint8
        )
        for gt_cls, gt_contour in zip(data_dict["gt_cls"], data_dict["gt_contour"]):
            cv2.drawContours(contour_mask, [gt_contour], -1, gt_cls, contour_width)
        np.save(data_dict["gt_contour_path"], contour_mask)

        # filled mask
        filled_mask = np.zeros(
            (len(general["categories"]), data_dict["img_height"], data_dict["img_width"]),
            dtype=np.uint8
        )
        for gt_cls, gt_contour in zip(data_dict["gt_cls"], data_dict["gt_contour"]):
            cv2.fillPoly(filled_mask[gt_cls], [gt_contour], 1)
        np.save(data_dict["gt_filled_path"], filled_mask)

        # copy image
        shutil.copy(os.path.join(img_folder, os.path.basename(data_dict["img_path"])), save_folder)
    
    # save
    for data_dict in general["data"]:
        data_dict.pop("gt_contour")
        #data_dict["gt_contour"] = [gt_contour.tolist() for gt_contour in data_dict["gt_contour"]]
    with open(os.path.join(save_folder, "general.json"), 'w') as f:
        json.dump(general, f, indent=4)
