from typing import Literal

import numpy as np

from ..format_conversion import BoxConvert


def get_iou(xmin1: int, ymin1: int, xmax1: int, ymax1: int, xmin2: int, ymin2: int, xmax2: int, ymax2: int) -> float:
    inter = max(0, min(ymax1, ymax2) - max(ymin1, ymin2)) * max(0, min(xmax1, xmax2) - max(xmin1, xmin2))
    areaA = (ymax1 - ymin1) * (xmax1 - xmin1)
    areaB = (ymax2 - ymin2) * (xmax2 - xmin2)
    return inter / (areaA + areaB - inter)


def nms_filter(bboxes: np.ndarray, src_type: Literal["voc", "yolo", "coco"], threshold=0.3):
    """
    Args:
        bboxes (np.ndarray): shape=(N, 4)
        src_type (str) 
        threshold (float) 
    Returns:
        results (np.ndarray): shape=(M, 4)
    """
    alive = set(range(len(bboxes)))
    results = []
    while len(alive) >= 2:
        min_alive = min(alive)
        results.append(min_alive)
        xmin1, ymin1, xmax1, ymax1 = BoxConvert.any2voc(
            src_type,
            bboxes[min_alive][0],
            bboxes[min_alive][1],
            bboxes[min_alive][2],
            bboxes[min_alive][3],
            width_for_yolo = 1000,
            height_for_yolo = 1000
        )
        alive.remove(min_alive)
        for idx in alive.copy():
            xmin2, ymin2, xmax2, ymax2 = BoxConvert.any2voc(
                src_type,
                bboxes[idx][0],
                bboxes[idx][1],
                bboxes[idx][2],
                bboxes[idx][3],
                width_for_yolo = 1000,
                height_for_yolo = 1000
            )
            iou  = get_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
            if iou >= threshold:
                alive.remove(idx)
    if len(alive)==1:
        results.append(alive.pop())
    return np.array(results)