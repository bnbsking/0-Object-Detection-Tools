import copy
import json
import os

import numpy as np

from ..detection import DetectionActiveLearningByHFlip
from ..utils.detection.augmentation import horizontal_flip


class InstanceSegmentationActiveLearningByHFlip(DetectionActiveLearningByHFlip):
    def __init__(
            self,
            pred_path_1: str,
            pred_path_2: str,
            save_path: str,
            lambda_: float = 1.0
        ):
        """
        lambda_ is the weight of mask consistency over instance consistency
        """
        pred1 = json.load(open(pred_path_1, 'r'))
        pred2 = json.load(open(pred_path_2, 'r'))
        pred2 = horizontal_flip(pred2)
        self.format_consistency_check(pred1, pred2)

        pred = copy.deepcopy(pred1)
        for i, (data_dict1, data_dict2) in enumerate(zip(pred1["data"], pred2["data"])):
            instance_loss = self.horizontal_consistency_loss(
                data_dict1["pd_boxes"],
                data_dict1["pd_probs"],
                data_dict2["pd_boxes"],
                data_dict2["pd_probs"]
            )
            mask_loss = self.mask_consistency_loss(
                data_dict1["pd_filled_path"],
                data_dict2["pd_filled_path"]
            )
            pred["data"][i]["loss"] = instance_loss + mask_loss * lambda_

        pred["data"] = sorted(pred["data"], key=lambda x: x["loss"], reverse=True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(pred, open(save_path, 'w'), indent=4)
    
    def mask_consistency_loss(
            self,
            pd_filled_path1: str,
            pd_filled_path2: str,
        ) -> float:
        mask1 = np.load(pd_filled_path1, allow_pickle=True)
        mask2 = np.load(pd_filled_path2, allow_pickle=True)

        cross_entropy = - mask1 * np.log(mask2 + 1e-10) \
            - (1 - mask1) * np.log(1 - mask2 + 1e-10) \
            - mask2 * np.log(mask1 + 1e-10) \
            - (1 - mask2) * np.log(1 - mask1 + 1e-10)
        
        return float(cross_entropy.mean())
    

class SemanticSegmentationActiveLearning:
    def __init__(self, pred_path: str, save_path: str, loss_name: str = "entropy"):
        pred = json.load(open(pred_path, 'r'))
        loss_func = getattr(self, loss_name)

        for data_dict in pred["data"]:
            data_dict["loss"] = loss_func(data_dict["pd_filled_path"])

        pred["data"] = sorted(pred["data"], key=lambda x: x["loss"], reverse=True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(pred, open(save_path, 'w'), indent=4)
    
    def entropy(self, pd_filled_path: str) -> float:
        mask = np.load(pd_filled_path, allow_pickle=True)
        return float((mask * np.log(mask + 1e-10)).mean())
