import copy
import json
from math import log
import os
from typing import Dict, List, Union

from ..utils.detection.augmentation import horizontal_flip
from ..utils.detection.tools import get_iou


class DetectionActiveLearningByHFlip:
    def __init__(self, pred_path_1: str, pred_path_2: str, save_path: str):
        pred1 = json.load(open(pred_path_1, 'r'))
        pred2 = json.load(open(pred_path_2, 'r'))
        pred2 = horizontal_flip(pred2)
        self.format_consistency_check(pred1, pred2)

        pred = copy.deepcopy(pred1)        
        for i, (data_dict1, data_dict2) in enumerate(zip(pred1["data"], pred2["data"])):
            pred["data"][i]["loss"] = self.horizontal_consistency_loss(
                data_dict1["pd_boxes"],
                data_dict1["pd_probs"],
                data_dict2["pd_boxes"],
                data_dict2["pd_probs"]
            )

        pred["data"] = sorted(pred["data"], key=lambda x: x["loss"], reverse=True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(pred, open(save_path, 'w'), indent=4)
    
    def format_consistency_check(self, pred1: Dict, pred2: Dict):
        assert pred1["categories"] == pred2["categories"]
        assert len(pred1["data"]) == len(pred2["data"])
        for data_dict1, data_dict2 in zip(pred1["data"], pred2["data"]):
            assert os.path.basename(data_dict1["img_path"]) \
                == os.path.basename(data_dict2["img_path"])
            assert type(data_dict1["gt_cls"]) == type(data_dict2["gt_cls"])

    def cross_entropy(self, probs1: List[float], probs2: List[float]) -> float:
        return sum(-p1 * log(p2 + 1e-10) for p1, p2 in zip(probs1, probs2))\
            / len(probs1)

    def horizontal_consistency_loss(
            self,
            pd_boxes1: List[List[int]],
            pd_probs1: List[float],
            pd_boxes2: List[List[int]],
            pd_probs2: List[float],
            iou_threshold: float = 0.5,
            lambda_: float = 1.0
        ) -> float:
        """
        Args:
            pd_boxes1: length=boxes1, in [xmin, ymin, xmax, ymax]
            pd_probs1: length=boxes1, probabilities of num_classes
            pd_boxes2: length=boxes2, in [xmin, ymin, xmax, ymax]
            pd_probs2: length=boxes2, probabilities of num_classes
            iou_threshold: threshold for matching
            lambda_: weight of unmatched loss
        Notes:
            matched_loss = (1 - iou) * mean_cross_entropy
            unmatched_loss = -log(background_prob)
            total_loss = matched_loss + unmatched_loss * lambda
        """
        loss = 0

        pds = [list(tup) for tup in zip([0] * len(pd_boxes2), pd_boxes2, pd_probs2)]  # (iou, box, prob)
        for box1, probs1 in zip(pd_boxes1, pd_probs1):
            xmin1, ymin1, xmax1, ymax1 = box1
            
            for i in range(len(pds)):
                _, (xmin2, ymin2, xmax2, ymax2), _ = pds[i]
                iou = get_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                pds[i][0] = iou

            pds.sort()
            if len(pds) and pds[-1][0] >= iou_threshold:  # matched loss
                iou, _, probs2 = pds.pop()
                loss += (1 - iou) * self.cross_entropy(probs1, probs2)
            else:  # unmatched loss 1
                loss += -log(probs1[0] + 1e-10) * lambda_
        
        # unmatched loss 2
        for _, _, probs2 in pds:
            loss += -log(probs2[0] + 1e-10) * lambda_
        
        return loss        
