from collections import Counter
import copy
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..detection.detection_confusion_matrix import ConfusionMatrix


class BaseMetricsPipeline:
    def __init__(
            self,
            num_classes: int,
            labels: List,
            predictions: List,
            func_dicts: List[Dict],
            save_path: str
        ):
        self.num_classes = num_classes
        self.labels = labels
        self.predictions = predictions
        self.func_dicts = func_dicts
        self.save_path = save_path

        self.metrics = {}
    
    def _deserialize(self, data: Dict):
        if isinstance(data, dict):
            return {k: self._deserialize(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def run(self) -> Dict:
        for func_dict in self.func_dicts:
            self.metrics[func_dict["log_name"]] = \
                getattr(self, func_dict["func_name"])(**func_dict["func_args"])
        with open(self.save_path, "w") as f:
            json.dump(self._deserialize(self.metrics), f, indent=4)
        return self.metrics


class DetectionMetricsPipeline(BaseMetricsPipeline):
    def __init__(
            self,
            num_classes: int,
            labels: List[np.ndarray],
            predictions: List[np.ndarray],
            func_dicts: List[Dict],
            save_path: str
        ):
        """
        Given basic arguments, self.run iterates func_dicts and save all metrics as results.
        Args:
            num_classes (int): number of classes
            labels (List[np.ndarray]): length is the number of images. each numpy has shape (N, 5).
                N is the number of ground truth, and 5 refers to (cid, xmin, ymin, xmax, ymax)
            predictions (List[np.ndarray]): length is the number of images. each numpy has shape (M, 5).
                M is the number of predictions, and 6 refers to (xmin, ymin, xmax, ymax, conf, cid)
            func_dicts (List[Dict]): length is the number of metrics function.
                each dict has the format {"func_name": str, "args": Dict, "log_name": str}.
                self.run saves the output in self.metrics, where log_name is key and output is value
            save_path (str): path to save the result as json
        """
        super().__init__(num_classes, labels, predictions, func_dicts, save_path)

        self.gt_class_cnts = self._get_gt_class_cnts(num_classes, labels)
        self.metrics = {}

    def _get_gt_class_cnts(self, num_classes: int, labels: List[np.ndarray]):
        gt_class_cnts = [0] * num_classes
        for label in labels:
            for i in range(len(label)):
                gt_class_cnts[label[i][0]] += 1
        return gt_class_cnts

    def get_pr_curves(self, k: int = 101) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(self.num_classes)
        ]

        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # get confusion of the threshold
            confusion = np.zeros(
                (self.num_classes + 1, self.num_classes + 1)
            )  # (i, j) = (pd, gt)
            for label, predictions in zip(self.labels, self.predictions):
                img_confusion = ConfusionMatrix(
                    self.num_classes,
                    CONF_THRESHOLD = threshold,
                    IOU_THRESHOLD = 0.5
                )
                img_confusion.process_batch(predictions, label)
                confusion += img_confusion.get_confusion()
            
            # update pr curve at the threshold from confusion
            row_sum = confusion.sum(axis=1)
            col_sum = confusion.sum(axis=0)
            for cid in range(self.num_classes):
                pr_curves[cid]["precision"][i] = confusion[cid][cid] / col_sum[cid] if col_sum[cid] else 0
                pr_curves[cid]["recall"][i] = confusion[cid][cid] / row_sum[cid] if row_sum[cid] else 0

        return pr_curves

    def get_refine_pr_curves(self, pr_curves_key: str = "pr_curves") -> List[Dict[str, List[float]]]:
        """
        sorted by recall, and enhance precision by next element reversely
        Args:
            pr_curves_key (str): get pr_curves from self.metrics and refine it.
        Dependency:
            you must call self.get_pr_curves in advance
        """
        refine_pr_curves = [{} for _ in range(self.num_classes)]
        pr_curves = copy.deepcopy(self.metrics[pr_curves_key])
        for cid in range(self.num_classes):
            recall_arr = pr_curves[cid]["recall"].copy()
            precision_arr = pr_curves[cid]["precision"].copy()
            zip_arr = sorted(zip(recall_arr, precision_arr))
            recall_arr, precision_arr = zip(*zip_arr)
            recall_arr, precision_arr = list(recall_arr), list(precision_arr)
            for i in range(1, len(precision_arr)):
                precision_arr[-1-i] = max(precision_arr[-1-i], precision_arr[-i])
            refine_pr_curves[cid]["refine_recall"] = recall_arr
            refine_pr_curves[cid]["refine_precision"] = precision_arr
        return refine_pr_curves
    
    def get_ap_list(self, refine_pr_curves_key: str = "refine_pr_curves") -> List[float]:
        """
        Args:
            refine_pr_curves_key (str): get refine_pr_curves from self.metrics and compute aps
        Dependency:
            you must call self.get_refine_pr_curves in advance
        """
        refine_pr_curves = self.metrics[refine_pr_curves_key]
        k_val = len(refine_pr_curves[0]["refine_precision"])  # 101
        ap_list = []
        for cid in range(self.num_classes):
            ap = 0
            for i in range(k_val - 1):
                ap += refine_pr_curves[cid]["refine_precision"][i] * \
                    (refine_pr_curves[cid]["refine_recall"][i+1] - refine_pr_curves[cid]["refine_recall"][i])
            ap_list.append(round(ap,3))
        return ap_list
    
    def get_map(self, ap_list_key: str) -> float:
        """
        Args:
            ap_list_key (str): get ap_list from self.metrics and compute map
        Dependency:
            you must call self.get_aps in advance
        """
        ap_list = self.metrics[ap_list_key]
        return round(sum(ap_list) / self.num_classes, 3)
    
    def get_wmap(self, ap_list_key: str) -> float:
        """
        Args:
            ap_list_key (str): get ap_list from self.metrics and compute wmap
        Dependency:
            you must call self.get_aps in advance
        """
        ap_list = self.metrics[ap_list_key]
        return round(sum(ap * cnt for ap, cnt in zip(ap_list, self.gt_class_cnts)) \
                / sum(self.gt_class_cnts), 3)
    
    def get_best_threshold(self, strategy: str = "f1", **kwargs) -> float:
        """
        get best threshold by some strategy
        Args:
            strategy (str): currently support "f1" or "precision" only
        Returns:
            best_threshold (float)
        Dependency:
            you must call self.get_pr_curves in advance
        """
        if strategy in {"f1", "precision"}:
            if strategy == "f1":
                score_func = lambda precision, recall: \
                    2 * precision * recall / (precision + recall + 1e-10)
            elif strategy == "precision":
                score_func = lambda precision, recall: \
                    precision if recall >= 0.5 else 0
            pr_curves_key = kwargs["pr_curves_key"]

            pr_curves = self.metrics[pr_curves_key]
            num_classes = len(pr_curves)
            k_val = len(pr_curves[0]["precision"])
            thresholds = np.linspace(0, 1, k_val)  # 101
            weighted_score = [0] * len(thresholds)
            for cid in range(num_classes):
                for i, (precision, recall) in enumerate(
                        zip(pr_curves[cid]["precision"], pr_curves[cid]["recall"])
                    ):
                    score = score_func(precision, recall)
                    weighted_score[i] += score * self.gt_class_cnts[cid] / sum(self.gt_class_cnts)
            _, best_threshold = max(zip(weighted_score, thresholds))
            return best_threshold

    def get_confusion(
            self,
            threshold: float = 0.5,
            threshold_key: str = ""
        ) -> np.ndarray:
        """
        Args:
            threshold (float)
            threshold_key (str): if specified, use self.metrics[threshold_key] instead of threshold
        Returns:
            confusion (np.ndarray[int]): shape=(num_classes+1, num_classes+1).
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        confusion = np.zeros((self.num_classes + 1, self.num_classes + 1))  # row: gt, col: pd
        for labels, predictions in zip(self.labels, self.predictions):
            cm = ConfusionMatrix(
                self.num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5
            )
            cm.process_batch(predictions, labels)
            confusion += cm.get_confusion()
        return confusion
    
    def get_confusion_with_img_indices(
            self,
            threshold: float = 0.5,
            threshold_key: str = ""
        ) -> List[List[Counter[int, int]]]:
        """
        Args:
            threshold (float)
            threshold_key (str): if specified, use self.metrics[threshold_key] instead of threshold
        Returns:
            confusion_with_img_indices (List[List[Counter[int, int]]]):
                shape=(num_classes+1, num_classes+1). each grid is Counters (img_idx -> cnts) 
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        confusion_with_img_indices = [
            [Counter() for _ in range(self.num_classes + 1)] for _ in range(self.num_classes + 1)
        ]
        for img_idx, (labels, predictions) in enumerate(zip(self.labels, self.predictions)):
            cm = ConfusionMatrix(
                self.num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5,
                img_idx=img_idx
            )
            cm.process_batch(predictions, labels)
            single_confusion_with_img_indices = cm.get_confusion_with_img_indices()
            for i in range(self.num_classes + 1):
                for j in range(self.num_classes + 1):
                    confusion_with_img_indices[i][j] += single_confusion_with_img_indices[i][j]
        return confusion_with_img_indices
    
    def get_confusion_axis_norm(self, confusion_key: str, axis: int) -> np.ndarray:
        """
        Args:
            axis (int): either 0 (col, precision) or 1 (row, recall)
        Returns:
            confusion_axis_norm (np.ndarray[float]): shape=(num_classes+1, num_classes+1).
        Dependency:
            you must call self.confusion in advance
        """
        confusion_axis_norm = self.metrics[confusion_key].copy()
        axis_sum = confusion_axis_norm.sum(axis=axis)
        for i in range(len(confusion_axis_norm)):
            if axis == 0:
                confusion_axis_norm[:, i] /= axis_sum[i]
            elif axis == 1:
                confusion_axis_norm[i, :] /= axis_sum[i]
        return confusion_axis_norm