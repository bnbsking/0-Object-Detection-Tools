import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]  # background first


def show_npy(
        img_path: str,
        npy_path: str,
        categories: List[str],
        save_path: Optional[str] = None,
        width: int = 4
    ):
    image = cv2.imread(img_path)
    segmentation_mask = np.load(npy_path, allow_pickle=True)

    # add segmentation to image
    seg_img = image.copy()
    for category_id in np.unique(segmentation_mask):
        if category_id > 0:
            color = colors[category_id % len(colors)]
            indices_x, indices_y = np.where(segmentation_mask == category_id)
            for x, y in zip(indices_x, indices_y):
                xlb = max(0, x - width)
                xub = min(segmentation_mask.shape[1], x + width)
                ylb = max(0, y - width)
                yub = min(segmentation_mask.shape[0], y + width)
                for channel in range(3):
                    seg_img[ylb: yub, xlb: xub, channel] = color[channel] * 255
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("raw image")
    plt.imshow(image[:, :, ::-1])
    plt.grid('on')

    plt.subplot(1, 2, 2)
    plt.title("segmentation")
    plt.imshow(seg_img)
    for i in range(len(categories)):
        plt.plot([], [], color=colors[i+1], label=categories[i])
    plt.legend(labels=categories)
    plt.grid('on')

    plt.show()
