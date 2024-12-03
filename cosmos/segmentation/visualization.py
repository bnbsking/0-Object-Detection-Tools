import os
import json
from typing import Callable, Dict, List, Literal, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]  # background first


def show_img_and_seg(
        img: np.ndarray,
        img_seg: np.ndarray,
        categories: List[str],
        save_path: str
    ):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("raw image")
    plt.imshow(img[:, :, ::-1])
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.title("segmentation")
    plt.imshow(img_seg)
    for i in range(len(categories)):
        plt.plot([], [], color=colors[i+1], label=categories[i])
    plt.legend(labels=categories)
    plt.grid('on')

    plt.savefig(save_path)
    plt.show()


def show_semantic_mask(
        img_path: str,
        npy_path: str,
        categories: List[str],
        save_path: Optional[str] = None,
        width: int = 4
    ):
    """
    npy shape=(h, w) and values are in {0, 1, 2, ..., max_category_id - 1}
    """
    img = cv2.imread(img_path)
    mask = np.load(npy_path, allow_pickle=True)

    # add segmentation to image
    img_seg = img.copy()
    for category_id in np.unique(mask):
        if category_id > 0:
            color = colors[category_id % len(colors)]
            indices_y, indices_x = np.where(mask == category_id)
            for x, y in zip(indices_x, indices_y):
                xlb = max(0, x - width)
                xub = min(mask.shape[1], x + width)
                ylb = max(0, y - width)
                yub = min(mask.shape[0], y + width)
                for channel in range(3):
                    img_seg[ylb: yub, xlb: xub, channel] = color[channel] * 255
    
    show_img_and_seg(img, img_seg, categories, save_path)


def show_instance_mask(
        img_path: str,
        npy_path: str,
        categories: List[str],
        save_path: Optional[str] = None,
        width: int = 4
    ):
    """
    npy shape=(category, h, w) and values are in {0, 1, 2, ..., max_instance_id_of_the_category}
    """
    img = cv2.imread(img_path)
    mask = np.load(npy_path, allow_pickle=True)

    # add segmentation to image
    img_seg = img.copy()
    for category_id in range(1, mask.shape[0]):
        color = colors[category_id % len(colors)]
        for instance_id in range(1, mask.shape[1]):
            indices_y, indices_x = np.where(mask[category_id] == instance_id)
            for x, y in zip(indices_x, indices_y):
                xlb = max(0, x - width)
                xub = min(mask.shape[2], x + width)
                ylb = max(0, y - width)
                yub = min(mask.shape[1], y + width)
                for channel in range(3):
                    img_seg[ylb: yub, xlb: xub, channel] = color[channel] * 255

    show_img_and_seg(img, img_seg, categories, save_path)


def show_general(
        folder_path: str,
        img_name: str,
        task: Literal["semantic", "instance"],
        save_folder: str = ".tmp"
    ):
    img_path = os.path.join(folder_path, img_name)
    contour_path = os.path.join(folder_path, "contour_" + img_name.replace(".jpg", ".npy"))
    filled_path = os.path.join(folder_path, "filled_" + img_name.replace(".jpg", ".npy"))
    categories = json.load(open(os.path.join(folder_path, "categories.json"), 'r'))[1:]
    
    show_semantic_mask(
        img_path,
        contour_path,
        categories,
        os.path.join(save_folder, "contour_" + img_name)
    )
    
    show_func = show_semantic_mask if task == "semantic" else show_instance_mask
    show_func(
        img_path,
        filled_path,
        categories,
        os.path.join(save_folder, "filled_" + img_name)
    )