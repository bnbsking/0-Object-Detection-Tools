from abc import abstractmethod
from collections import Counter
import copy
import json
import os
from tqdm import tqdm
from typing import Dict, List, Union

import numpy as np
from sklearn import metrics as skm

from ..detection.confusion_matrix import DetectionConfusionMatrix, SegmentationConfusionMatrix


class BaseMetricsPipeline:
    def __init__(
            self,
            num_classes: int,
            labels: List,
            predictions: List,
            func_dicts: List[Dict],
            save_path: str,
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
        elif isinstance(data, list):
            return [self._deserialize(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def run(self) -> Dict:
        for func_dict in self.func_dicts:
            self.metrics[func_dict["log_name"]] = \
                getattr(self, func_dict["func_name"])(**func_dict["func_args"])
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w") as f:
            metrics = self._deserialize(self.metrics)
            json.dump(metrics, f, indent=4)
        return self.metrics


class CommonMetricsPipeline(BaseMetricsPipeline):
    """
    This class provides some common metrics for each tasks
    """
    gt_class_cnts: List[int]

    @abstractmethod
    def get_pr_curves(self, k: int = 101) -> List[Dict[str, List[float]]]:
        """
        Args:
            k (int): the granularity of thresholds. i.e. Scanning thresholds within np.linspace(0,1,k) 
        Returns:
            pr_curves (List[Dict]): for example, [
                    {
                        "precision": [0.] * k,
                        "recall": [0.] * k,
                    } for _ in range(n)
                ]  # n = self.num_classses or self.num_classes - 1
        """
        raise NotImplementedError

    def get_refine_pr_curves(self, pr_curves_key: str = "pr_curves") -> List[Dict[str, List[float]]]:
        """
        sorted by recall, and enhance precision by next element reversely
        Args:
            pr_curves_key (str): get pr_curves from self.metrics and refine it.
        Dependency:
            you must call self.get_pr_curves in advance
        """
        pr_curves = copy.deepcopy(self.metrics[pr_curves_key])
        refine_pr_curves = [{} for _ in range(len(pr_curves))]
        for i in range(len(pr_curves)):
            recall_arr = pr_curves[i]["recall"].copy()
            precision_arr = pr_curves[i]["precision"].copy()
            zip_arr = sorted(zip(recall_arr, precision_arr))
            recall_arr, precision_arr = zip(*zip_arr)
            recall_arr, precision_arr = list(recall_arr), list(precision_arr)
            for j in range(1, len(precision_arr)):
                precision_arr[-1-j] = max(precision_arr[-1-j], precision_arr[-j])
            refine_pr_curves[i]["refine_recall"] = recall_arr
            refine_pr_curves[i]["refine_precision"] = precision_arr
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
        for i in range(len(refine_pr_curves)):
            ap = 0
            for j in range(k_val - 1):
                ap += refine_pr_curves[i]["refine_precision"][j] * \
                    (refine_pr_curves[i]["refine_recall"][j+1] - refine_pr_curves[i]["refine_recall"][j])
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
        return round(sum(ap_list) / len(ap_list), 3)
    
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
        Notes:
            For classification does not have background, the return value is meaningless
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
            k_val = len(pr_curves[0]["precision"])
            thresholds = np.linspace(0, 1, k_val)  # 101
            weighted_score = [0] * len(thresholds)
            for i in range(len(pr_curves)):
                for j, (precision, recall) in enumerate(
                        zip(pr_curves[i]["precision"], pr_curves[i]["recall"])
                    ):
                    score = score_func(precision, recall)
                    weighted_score[j] += score * self.gt_class_cnts[i] / sum(self.gt_class_cnts)
            _, best_threshold = max(zip(weighted_score, thresholds))
            return best_threshold
        else:
            return 0.5

    @abstractmethod
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
            confusion (np.ndarray[int]): shape=(num_classes, num_classes)
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        raise NotImplementedError
    
    def get_confusion_axis_norm(self, confusion_key: str, axis: int) -> np.ndarray:
        """
        Args:
            axis (int): either 0 (col, precision) or 1 (row, recall)
        Returns:
            confusion_axis_norm (np.ndarray[float]): shape same as input confusion
        Dependency:
            you must call self.confusion in advance
        """
        confusion_axis_norm = self.metrics[confusion_key].copy()
        axis_sum = confusion_axis_norm.sum(axis=axis)
        for i in range(len(confusion_axis_norm)):
            if axis == 0:
                confusion_axis_norm[:, i] /= (axis_sum[i] + 1e-10)
            elif axis == 1:
                confusion_axis_norm[i, :] /= (axis_sum[i] + 1e-10)
        return confusion_axis_norm


class ClassificationMetricsPipeline(CommonMetricsPipeline):
    def __init__(
            self,
            num_classes: int,
            labels: Union[List[int], List[List[int]]],
            predictions: np.ndarray,
            func_dicts: List[Dict],
            save_path: str,
            start_idx: int
        ):
        """
        Given basic arguments, self.run iterates func_dicts and save all metrics as results.
        Args:
            num_classes (int): number of classes (includes background if exist).
            labels (np.ndarray):
                For single label, shape=(data,)
                For multi label, shape=(data, multi-label-dim).
            predictions (np.ndarray):
                For single label, shape=(data, num_classes). If background exist, it must be zeroth category. 
                For multi label, shape=(data, multi-label-dim, 2)
            func_dicts (List[Dict]): length is the number of metrics function.
                each dict has the format {"func_name": str, "args": Dict, "log_name": str}.
                self.run saves the output in self.metrics, where log_name is key and output is value
            save_path (str): path to save the result as json
            start_idx (int): be 0 or 1 (has background)
        """
        super().__init__(num_classes, labels, predictions, func_dicts, save_path)
        self.single_label = len(labels.shape) == 1
        self.start_idx = self.single_label and start_idx
        self.gt_class_cnts = self._get_gt_class_cnts()

    def _get_gt_class_cnts(self) -> List[int]:
        gt_class_cnts = [0] * (self.num_classes - self.start_idx)
        if self.single_label:
            for label in self.labels:
                gt_class_cnts[label - self.start_idx] += 1
        else:
            for label_list in self.labels:  # multilabel returns positive count only
                for cls_idx, is_positive in enumerate(label_list):
                    gt_class_cnts[cls_idx] += is_positive
        return gt_class_cnts

    def get_pr_curves(self, k: int = 101) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(self.num_classes - self.start_idx)
        ]

        for i, threshold in enumerate(np.linspace(0, 1, k)):
            for j in range(self.start_idx, self.num_classes):
                if self.single_label:
                    gt_cls = self.labels
                    pd_cls = (self.predictions[:, j] >= threshold).astype(np.int32)
                else:
                    gt_cls = self.labels[:, j]
                    pd_cls = (self.predictions[:, j, 1] >= threshold).astype(np.int32)
                precision = skm.precision_score(gt_cls, pd_cls, zero_division=0.0)
                recall = skm.recall_score(gt_cls, pd_cls, zero_division=0.0)
                pr_curves[j - self.start_idx]["precision"][i] = precision
                pr_curves[j - self.start_idx]["recall"][i] = recall

        return pr_curves
    
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
            confusion (np.ndarray[int]): shape=(num_classes, num_classes).
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        Notes:
            For multi-class classification does not have background, threshold is meaningless.
        """
        if self.single_label and self.start_idx==0 and self.num_classes > 2:  # multiclass without background
            gt_cls = self.labels
            pd_cls = self.predictions.argmax(axis=1)
        elif self.single_label:  # binary or multiclass with background
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels
            pd_cls = np.where(
                    self.predictions[:, 0] < threshold,
                    0,
                    self.predictions[:, 1:].argmax(axis=1) + 1
                )
        else:  # multilabel
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels.reshape(-1)
            pd_cls = (self.predictions[:, :, 1] >= threshold).reshape(-1).astype(np.int32)
        
        confusion = skm.confusion_matrix(gt_cls, pd_cls, labels=list(range(self.num_classes)))
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
                shape=(num_classes, num_classes). each grid is counter of image indices
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        Notes:
            For multi-class classification does not have background, threshold is meaningless.
        """
        if self.single_label and self.start_idx==0 and self.num_classes > 2:  # multiclass without background
            gt_cls = self.labels
            pd_cls = self.predictions.argmax(axis=1)
        elif self.single_label:  # binary or multiclass with background
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels
            pd_cls = np.where(
                    self.predictions[:, 0] > threshold,
                    0,
                    self.predictions[:, 1:].argmax(axis=1) + 1
                )
        else:  # multilabel
            threshold = self.metrics[threshold_key] if threshold_key else threshold
            gt_cls = self.labels.reshape(-1)
            pd_cls = (self.predictions[:, :, 1] >= threshold).reshape(-1).astype(np.int32)
        
        confusion_with_img_indices = [
            [Counter() for _ in range(self.num_classes)] for _ in range(self.num_classes)
        ]
        dataset_length = len(self.labels)
        for idx, (gtc, pdc) in enumerate(zip(gt_cls, pd_cls)):
            confusion_with_img_indices[gtc][pdc][idx % dataset_length] += 1
        return confusion_with_img_indices


class DetectionMetricsPipeline(CommonMetricsPipeline):
    def __init__(
            self,
            num_classes: int,
            labels: List[np.ndarray],
            predictions: List[np.ndarray],
            func_dicts: List[Dict],
            save_path: str,
            **kwargs
        ):
        """
        Given basic arguments, self.run iterates func_dicts and save all metrics as results.
        Args:
            num_classes (int): number of classes, where the first element is "__background__".
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

    def _get_gt_class_cnts(self, num_classes: int, labels: List[np.ndarray]) -> List[int]:
        gt_class_cnts = [0] * (num_classes - 1)
        for label in labels:
            for i in range(len(label)):
                gt_class_cnts[label[i][0] - 1] += 1
        return gt_class_cnts

    def get_pr_curves(self, k: int = 101) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(self.num_classes - 1)
        ]

        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # get confusion of the threshold
            confusion = np.zeros(
                (self.num_classes, self.num_classes)
            )  # (i, j) = (gt, pd)

            for labels, predictions in zip(self.labels, self.predictions):
                img_confusion = DetectionConfusionMatrix(
                    self.num_classes,
                    CONF_THRESHOLD = threshold,
                    IOU_THRESHOLD = 0.5
                )
                img_confusion.process_batch(predictions, labels)
                confusion += img_confusion.get_confusion()
            
            # update pr curve at the threshold from confusion
            row_sum = confusion.sum(axis=1)
            col_sum = confusion.sum(axis=0)
            for cid in range(1, self.num_classes):
                pr_curves[cid-1]["precision"][i] = confusion[cid][cid] / col_sum[cid] if col_sum[cid] else 0
                pr_curves[cid-1]["recall"][i] = confusion[cid][cid] / row_sum[cid] if row_sum[cid] else 0

        return pr_curves

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
            confusion (np.ndarray[int]): shape=(num_classes, num_classes).
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        confusion = np.zeros((self.num_classes, self.num_classes))  # row: gt, col: pd
        for labels, predictions in zip(self.labels, self.predictions):
            cm = DetectionConfusionMatrix(
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
                shape=(num_classes, num_classes). each grid is Counters (img_idx -> cnts) 
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        confusion_with_img_indices = [
            [Counter() for _ in range(self.num_classes)] for _ in range(self.num_classes)
        ]
        for img_idx, (labels, predictions) in enumerate(zip(self.labels, self.predictions)):
            cm = DetectionConfusionMatrix(
                self.num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5,
                img_idx=img_idx
            )
            cm.process_batch(predictions, labels)
            single_confusion_with_img_indices = cm.get_confusion_with_img_indices()
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    confusion_with_img_indices[i][j] += single_confusion_with_img_indices[i][j]
        return confusion_with_img_indices


class InstanceSegmentationMetricsPipeline(CommonMetricsPipeline):
    def __init__(
            self,
            num_classes: int,
            labels: List[np.ndarray],
            predictions: List[np.ndarray],
            func_dicts: List[Dict],
            save_path: str,
            **kwargs
        ):
        """
        Given basic arguments, self.run iterates func_dicts and save all metrics as results.
        Args:
            num_classes (int): number of classes, where the first element is "__background__".
            labels (List[Dict]): length is num of images. Each dictionary contains:
                - detection (np.array): shape=(labels_for_an_img, 5).
                - segmentation_path (str): path to the segmentation mask.
            predictions (List[Dict]): length is num of images. Each dictionary contains:
                - detection (np.array): shape=(num_boxes, 6).
                    Each prediction is represented as (xmin, ymin, xmax, ymax, conf, cid), where:
                    - xmin, ymin, xmax, ymax: Bounding box coordinates.
                    - conf: Confidence score of the prediction.
                    - cid: Class ID with the highest confidence score.
                - segmentation_path (str): path to the segmentation mask.
            func_dicts (List[Dict]): length is the number of metrics function.
                each dict has the format {"func_name": str, "args": Dict, "log_name": str}.
                self.run saves the output in self.metrics, where log_name is key and output is value
            save_path (str): path to save the result as json
        """
        super().__init__(num_classes, labels, predictions, func_dicts, save_path)
        self.gt_class_cnts = self._get_gt_class_cnts(num_classes, labels)
    
    def _get_gt_class_cnts(self, num_classes: int, labels: List[Dict]) -> List[int]:
        gt_class_cnts = [0] * (num_classes - 1)
        for label_dict in labels:
            label = label_dict["detection"]
            for i in range(len(label)):
                gt_class_cnts[label[i][0] - 1] += 1
        return gt_class_cnts
    
    def get_pr_curves(self, k: int = 101) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(self.num_classes - 1)
        ]

        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # get confusion of the threshold
            confusion = np.zeros(
                (self.num_classes, self.num_classes)
            )  # (i, j) = (gt, pd)
            
            for label_dict, prediction_dict in zip(self.labels, self.predictions):
                label_mask = np.load(label_dict["segmentation_path"], allow_pickle=True)
                prediction_mask = np.load(prediction_dict["segmentation_path"], allow_pickle=True)

                img_confusion = SegmentationConfusionMatrix(
                    self.num_classes,
                    CONF_THRESHOLD = threshold,
                    IOU_THRESHOLD = 0.5
                )
                img_confusion.process_batch(
                    prediction_dict["detection"],
                    label_dict["detection"],
                    prediction_mask,
                    label_mask
                )
                confusion += img_confusion.get_confusion()
            
            # update pr curve at the threshold from confusion
            row_sum = confusion.sum(axis=1)
            col_sum = confusion.sum(axis=0)
            for cid in range(1, self.num_classes):
                pr_curves[cid-1]["precision"][i] = confusion[cid][cid] / col_sum[cid] if col_sum[cid] else 0
                pr_curves[cid-1]["recall"][i] = confusion[cid][cid] / row_sum[cid] if row_sum[cid] else 0

        return pr_curves
    
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
            confusion (np.ndarray[int]): shape=(num_classes, num_classes).
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        confusion = np.zeros((self.num_classes, self.num_classes))  # row: gt, col: pd
        
        for label_dict, prediction_dict in zip(self.labels, self.predictions):
            label_mask = np.load(label_dict["segmentation_path"], allow_pickle=True)
            prediction_mask = np.load(prediction_dict["segmentation_path"], allow_pickle=True)
            cm = SegmentationConfusionMatrix(
                self.num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5
            )
            cm.process_batch(
                prediction_dict["detection"],
                label_dict["detection"],
                prediction_mask,
                label_mask
            )
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
                shape=(num_classes, num_classes). each grid is Counters (img_idx -> cnts) 
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        confusion_with_img_indices = [
            [Counter() for _ in range(self.num_classes)] for _ in range(self.num_classes)
        ]

        for img_idx, (label_dict, prediction_dict) in enumerate(zip(self.labels, self.predictions)):
            label_mask = np.load(label_dict["segmentation_path"], allow_pickle=True)
            prediction_mask = np.load(prediction_dict["segmentation_path"], allow_pickle=True)
            cm = SegmentationConfusionMatrix(
                self.num_classes,
                CONF_THRESHOLD=threshold,
                IOU_THRESHOLD=0.5,
                img_idx=img_idx
            )
            cm.process_batch(
                prediction_dict["detection"],
                label_dict["detection"],
                prediction_mask,
                label_mask
            )
            single_confusion_with_img_indices = cm.get_confusion_with_img_indices()
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    confusion_with_img_indices[i][j] += single_confusion_with_img_indices[i][j]
        
        return confusion_with_img_indices


class SemanticSegmentationMetricsPipeline(CommonMetricsPipeline):
    def __init__(
            self,
            num_classes: int,
            labels: List[np.ndarray],
            predictions: List[np.ndarray],
            func_dicts: List[Dict],
            save_path: str,
            **kwargs
        ):
        """
        Given basic arguments, self.run iterates func_dicts and save all metrics as results.
        Args:
            num_classes (int): number of classes, where the first element is "__background__".
            labels (List[Dict]): length is num of images. Each dictionary contains:
                - detection (np.array): empty
                - segmentation_path (str): path to the segmentation mask.
            predictions (List[Dict]): length is num of images. Each dictionary contains:
                - detection (np.array): empty
                - segmentation_path (str): path to the segmentation mask.
            func_dicts (List[Dict]): length is the number of metrics function.
                each dict has the format {"func_name": str, "args": Dict, "log_name": str}.
                self.run saves the output in self.metrics, where log_name is key and output is value
            save_path (str): path to save the result as json
        """
        super().__init__(num_classes, labels, predictions, func_dicts, save_path)
        self.gt_class_cnts = self._get_gt_class_cnts(num_classes, labels)
    
    def _get_gt_class_cnts(self, num_classes: int, labels: List[Dict]) -> List[int]:
        gt_class_cnts = [0] * (num_classes - 1)
        for label_dict in labels:
            label = np.load(label_dict["segmentation_path"], allow_pickle=True)
            for i in range(1, label.shape[0]):
                gt_class_cnts[i - 1] += int(np.sum(label[i]))
        return gt_class_cnts
    
    def get_pr_curves(self, k: int = 101) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(self.num_classes - 1)
        ]

        range_list = list(range(self.num_classes))
        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # update pr curve at the threshold from confusion
            confusion = np.zeros(
                (self.num_classes, self.num_classes)
            )  # (i, j) = (gt, pd)
            
            for label_dict, prediction_dict in zip(self.labels, self.predictions):
                label_mask = np.load(label_dict["segmentation_path"], allow_pickle=True)
                label = label_mask.argmax(axis=0)
                prediction_mask = np.load(prediction_dict["segmentation_path"], allow_pickle=True)
                prediction_argmax = prediction_mask.argmax(axis=0)
                prediction = np.where(
                    prediction_mask.max(axis=0) >= threshold, prediction_argmax, 0
                )
                confusion += skm.confusion_matrix(
                    label.reshape(-1),
                    prediction.reshape(-1),
                    labels = range_list
                )
            
            # update pr curve at the threshold from confusion
            row_sum = confusion.sum(axis=1)
            col_sum = confusion.sum(axis=0)
            for cid in range(1, self.num_classes):
                pr_curves[cid-1]["precision"][i] = confusion[cid][cid] / col_sum[cid] if col_sum[cid] else 0
                pr_curves[cid-1]["recall"][i] = confusion[cid][cid] / row_sum[cid] if row_sum[cid] else 0

        return pr_curves
    
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
            confusion (np.ndarray[int]): shape=(num_classes, num_classes).
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        range_list = list(range(self.num_classes))
        confusion = np.zeros((self.num_classes, self.num_classes))  # row: gt, col: pd
        
        for label_dict, prediction_dict in zip(self.labels, self.predictions):
            label_mask = np.load(label_dict["segmentation_path"], allow_pickle=True)
            label = label_mask.argmax(axis=0)
            prediction_mask = np.load(prediction_dict["segmentation_path"], allow_pickle=True)
            prediction_argmax = prediction_mask.argmax(axis=0)
            
            if self.num_classes == 2:
                prediction = np.where(
                    prediction_mask.max(axis=0) >= threshold, prediction_argmax, 0
                )
            else:
                prediction = prediction_argmax
            
            confusion += skm.confusion_matrix(
                label.reshape(-1),
                prediction.reshape(-1),
                labels = range_list
            )

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
                shape=(num_classes, num_classes). each grid is Counters (img_idx -> cnts) 
        Dependency:
            if threshold_key is specified, you must call self.get_best_threshold in advance
        """
        threshold = self.metrics[threshold_key] if threshold_key else threshold
        range_list = list(range(self.num_classes))
        confusion_with_img_indices = [
            [Counter() for _ in range(self.num_classes)] for _ in range(self.num_classes)
        ]

        for img_idx, (label_dict, prediction_dict) in enumerate(zip(self.labels, self.predictions)):
            label_mask = np.load(label_dict["segmentation_path"], allow_pickle=True)
            label = label_mask.argmax(axis=0)
            prediction_mask = np.load(prediction_dict["segmentation_path"], allow_pickle=True)
            prediction_argmax = prediction_mask.argmax(axis=0)
            
            if self.num_classes == 2:
                prediction = np.where(
                    prediction_mask.max(axis=0) >= threshold, prediction_argmax, 0
                )
            else:
                prediction = prediction_argmax
            
            confusion = skm.confusion_matrix(
                label.reshape(-1),
                prediction.reshape(-1),
                labels = range_list
            )

            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    confusion_with_img_indices[i][j] += Counter({img_idx: int(confusion[i][j])})
        
        return confusion_with_img_indices