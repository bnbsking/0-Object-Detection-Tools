from collections import defaultdict
import json
import os
import shutil

import cv2
import numpy as np


def coco_to_npy(img_folder: str, ant_path: str, save_folder: str):
    with open(ant_path, 'r') as f:
        coco = json.load(f)
    os.makedirs(save_folder, exist_ok=True)

    # Get categories and save to json
    max_cat_id = max(category['id'] for category in coco['categories'])
    categories = ["__background__"] * (max_cat_id + 1)
    for category in coco['categories']:
        categories[category['id']] = category['name']
    json.dump(categories, open(os.path.join(save_folder, "categories.json"), 'w'))

    # Get image_id to all info
    img_id_to_all = {}
    for img_dict in coco["images"]:
        img_id_to_all[img_dict["id"]] = {
            "file_name": img_dict["file_name"],
            "shape": (img_dict["height"], img_dict["width"]),
            "ant": []
        }
    for ant_dict in coco["annotations"]:
        # change segmentation (x1, y1, ...) into contour [(x1, y1), ...]
        contour = []
        for seg_list in ant_dict["segmentation"]:
            for i in range(len(seg_list) // 2):
                contour.append((seg_list[2 * i], seg_list[2 * i + 1]))
        contour = np.array(contour, dtype=np.int32)
        # collect ant
        img_id_to_all[ant_dict["image_id"]]["ant"].append(
            {
                "category_id": ant_dict["category_id"],
                "contour": contour,
            }
        )

    # Get contour_mask and filled_mask in npy format and save
    for img_info in img_id_to_all.values():
        contour_mask = np.zeros(img_info["shape"], dtype=np.uint8)
        filled_mask = np.zeros(img_info["shape"], dtype=np.uint8)
        for ant in img_info["ant"]:
            cv2.drawContours(contour_mask, [ant["contour"]], -1, ant["category_id"], 2)
            cv2.fillPoly(filled_mask, [ant["contour"]], ant["category_id"])
        
        np.save(
            os.path.join(save_folder, "contour_" + img_info["file_name"].replace(".jpg", ".npy")), 
            contour_mask
        )
        np.save(
            os.path.join(save_folder, "filled_" + img_info["file_name"].replace(".jpg", ".npy")), 
            filled_mask
        )
        # copy image
        shutil.copy(os.path.join(img_folder, img_info["file_name"]), os.path.join(save_folder, img_info["file_name"]))
