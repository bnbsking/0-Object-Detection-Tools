import importlib
import json
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]  # background first


def import_method_from_module(
        method_name: str,
        class_name: str = "FormatConvertAny2General",
        module_name: str = ".format_conversion"
    ) -> Callable:
    module = importlib.import_module(module_name, package=__package__)
    return getattr(getattr(module, class_name), method_name)


def stroke(patch: np.ndarray, seg_list: List[int], color = (1,1,1)):
    # seg_list: up, mid, down, upleft, upright, downleft, downright
    if not seg_list:
        patch[:3, :3, :] = 0, 0, 0
        patch[-3:, -3:, :] = 0, 0, 0
        for i in range(10):
            patch[i*2:i*2+2, 10-i-1:10-i, :] = 0, 0, 0
    else:
        if seg_list[0]:
            patch[0:3, :, :] = 0, 0, 0
        if seg_list[1]:
            patch[10-1:10+1, :, :] = 0, 0, 0        
        if seg_list[2]:
            patch[20-3:20, :, :] = 0, 0, 0
        if seg_list[3]:
            patch[:10, :3, :] = 0, 0, 0
        if seg_list[4]:
            patch[:10, 10-3:, :] = 0, 0, 0
        if seg_list[5]:
            patch[10:, :3, :] = 0, 0, 0
        if seg_list[6]:
            patch[10:, 10-3:,:] = 0, 0, 0
    B = np.array([[color for _ in range(20)] for _ in range(30)]).astype(float)
    B[5:25, 5:15, :] = patch
    return B


def get_img(digit: int, color = (1,1,1)):
    patch = np.array([[color for _ in range(10)] for _ in range(20)]).astype(float)
    seven_segment_display = {
            0: [1, 0, 1, 1, 1, 1, 1],
            1: [0, 0, 0, 1, 0, 1, 0],
            2: [1, 1, 1, 0, 1, 1, 0],
            3: [1, 1, 1, 0, 1, 0, 1],
            4: [0, 1, 0, 1, 1, 0, 1],
            5: [1, 1, 1, 1, 0, 0, 1],
            6: [1, 1, 1, 1, 0, 1, 1],
            7: [1, 0, 0, 0, 1, 0, 1],
            8: [1, 1, 1, 1, 1, 1, 1],
            9: [1, 1, 1, 1, 1, 0, 1],
            10: [],  # percent
        }
    return stroke(patch, seven_segment_display[digit], color)


def get_patch(unit_digit: int, tens_digit: int, color = (1, 1, 1)):
    A = np.array([[color for _ in range(60)] for _ in range(30)]).astype(float)
    A[:, :20, :] = get_img(unit_digit, color)
    A[:, 20:40, :] = get_img(tens_digit, color)
    A[:, 40:60, :] = get_img(10, color)
    return A


def show(
        class_list: str,
        data_dict: Dict,
        save_path: Optional[str] = None,
        box_width: int = 4,
        value_ratios: Tuple[int, int] = (1,1)
    ):
    class_list.pop(0) if class_list[0] == "__background__" else None
    img_raw = cv2.imread(data_dict["img_path"])[:, :, ::-1]/255

    # ground truth
    img_gt = img_raw.copy()
    boxes_gt = data_dict["gt_boxes"] 
    cids_gt = data_dict["gt_cls"]
    for (xmin, ymin, xmax, ymax), cid in zip(boxes_gt, cids_gt):
        img_gt[ymin-box_width:ymin+box_width, xmin:xmax, :] = colors[cid]
        img_gt[ymax-box_width:ymax+box_width, xmin:xmax, :] = colors[cid]
        img_gt[ymin:ymax, xmin-box_width:xmin+box_width, :] = colors[cid]
        img_gt[ymin:ymax, xmax-box_width:xmax+box_width, :] = colors[cid]
    
    # prediction
    img_pd = img_raw.copy()
    pd_probs = data_dict.get("pd_probs", [])
    pd_boxes = data_dict.get("pd_boxes", [])

    if pd_probs:
        pd_confs = np.array(pd_probs).max(axis=1)
        pd_cids = np.array(pd_probs).argmax(axis=1)
    else:
        pd_confs = []
        pd_cids = []

    for pd_conf, (xmin, ymin, xmax, ymax), pd_cid in sorted(zip(pd_confs, pd_boxes, pd_cids)):  # plot least conf first
        img_pd[ymin-box_width:ymin+box_width, xmin:xmax, :] = colors[pd_cid]
        img_pd[ymax-box_width:ymax+box_width, xmin:xmax, :] = colors[pd_cid]
        img_pd[ymin:ymax, xmin-box_width:xmin+box_width, :] = colors[pd_cid]
        img_pd[ymin:ymax, xmax-box_width:xmax+box_width, :] = colors[pd_cid]
        
        # confidence patches
        unit_digit, tens_digit = int(pd_conf * 10), int(pd_conf * 100) % 10
        P = get_patch(unit_digit, tens_digit, color=colors[pd_cid])
        (ph, pw, _), (rh, rw) = P.shape, value_ratios
        P = cv2.resize(P, (int(pw * rw), int(ph * rh)) )
        try:
            if ymin >= P.shape[0] and xmin + P.shape[1] < img_pd.shape[1]:  # upper bar - up
                img_pd[ymin - P.shape[0]:ymin, xmin:xmin + P.shape[1], :] = P
            elif ymax + P.shape[0] < img_pd.shape[0] and xmin + P.shape[1] < img_pd.shape[1]:  # down bar - down
                img_pd[ymax:ymax + P.shape[0], xmin:xmin + P.shape[1], :] = P
            elif ymin + P.shape[0] < img_pd.shape[0] and xmin + P.shape[1]<img_pd.shape[1]:
                img_pd[ymin:ymin+P.shape[0], xmin:xmin + P.shape[1], :] = P  # upper bar - down
            elif ymax + P.shape[0] > 0 and xmin + P.shape[1]<img_pd.shape[1]:  # down bar - up
                img_pd[ymax - P.shape[0]:ymax, xmin:xmin + P.shape[1], :] = P
        except:
            pass

    # plot
    fig = plt.figure(figsize=(20, 10))
    fig.set_facecolor("white")

    plt.subplot(1, 2, 1)
    plt.title("GT", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    for r, g, b in colors:
        c2hex = lambda c: hex(int(c * 255))[2:].zfill(2)
        plt.scatter([0], [0], c=f"#{c2hex(r)}{c2hex(g)}{c2hex(b)}")

    plt.legend(labels=class_list, fontsize=16)
    plt.imshow(img_gt)
    
    plt.subplot(1, 2, 2)
    plt.title("Pred", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.imshow(img_pd)
    
    plt.show()
    plt.savefig(save_path) if save_path else None
    plt.close()


def show_general(img_name: str, ant_path: str, save_path: Optional[str] = None):
    """
    Show an image with its ground truth (and prediction if it exists) in general format.
    Hint: Convert coco / voc / yolo to general first by `coco2general` / `voc2general` / `yolo2general`
    Args:
        img_name (str): name of target image to be shown
        ant_path (str): path to the general format
        save_path (Optional[str]): path to save the visualized result
    """
    with open(ant_path, "r", encoding="utf-8") as f:
        general = json.load(f)
        class_list = general["categories"]
    data_dict = next(data_dict for data_dict in general["data"] if os.path.basename(data_dict["img_path"])==img_name)
    show(class_list, data_dict, save_path)


def show_coco(
        img_name: str,
        img_folder: str,
        ant_path: str,
        save_folder: str = ".tmp",
        use_cache: bool = True,
    ):
    """
    Show an image with its ground truth in coco format.
    This func will convert coco format data into `general` format.
    Args:
        img_name (str): name of target image to be shown
        img_folder (str): path to the image folder
        ant_path (str): path to the coco label
        save_folder (Optional[str], optional): folder saves the conversion result and visualized output
        use_cache (bool, optional): if true, the conversion execute once only.
    """
    coco2general = import_method_from_module("coco2general")
    
    general_path = os.path.join(save_folder, ".tmp_general.json")
    if not use_cache or not os.path.exists(general_path):
        coco2general(img_folder, ant_path, general_path)
    save_path = os.path.join(save_folder, img_name)
    
    show_general(img_name, general_path, save_path)


def show_voc(
        img_name: str,
        img_path_list: List[str],
        ant_path_list: List[str],
        class_list: List[str],
        save_folder: str = ".tmp",
        use_cache: bool = True,
    ):
    """
    Show an image with its ground truth in voc format.
    This func will convert voc format data into `general` format.
    Args:
        img_name (str): name of target image to be shown
        img_path_list (List[str]): list of image path of the dataset
        ant_path_list (List[str]): list of annotation path of the dataset
        class_list (List[str]): list of class name
        save_folder (Optional[str], optional): folder saves the conversion result and visualized output
        use_cache (bool, optional): if true, the conversion execute once only.
    """
    voc2general = import_method_from_module("voc2general")
    
    general_path = os.path.join(save_folder, ".tmp_general.json")
    if not use_cache or not os.path.exists(general_path):
        voc2general(img_path_list, ant_path_list, class_list, general_path)
    save_path = os.path.join(save_folder, img_name)

    show_general(img_name, general_path, save_path)


def show_yolo(
        img_name: str,
        img_path_list: List[str],
        ant_path_list: List[str],
        class_list: List[str],
        save_folder: str = ".tmp",
        use_cache: bool = True,
    ):
    """
    Show an image with its ground truth in yolo format.
    This func will convert yolo format data into `general` format.
    Args:
        img_name (str): name of target image to be shown
        img_path_list (List[str]): list of image path of the dataset
        ant_path_list (List[str]): list of annotation path of the dataset
        class_list (List[str]): list of class name
        save_folder (Optional[str], optional): folder saves the conversion result and visualized output
        use_cache (bool, optional): if true, the conversion execute once only.
    """
    yolo2general = import_method_from_module("yolo2general")
    
    general_path = os.path.join(save_folder, ".tmp_general.json")
    if not use_cache or not os.path.exists(general_path):
        yolo2general(img_path_list, ant_path_list, class_list, general_path)
    save_path = os.path.join(save_folder, img_name)

    show_general(img_name, general_path, save_path)
