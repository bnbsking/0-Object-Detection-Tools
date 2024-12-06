"""
Reference: https://github.com/kaanakan/object_detection_confusion_matrix/blob/master/confusion_matrix.py
"""
from abc import abstractmethod
from collections import Counter
from typing import Dict, List, Optional

import numpy as np


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class BaseConfusionMatrix:
    def __init__(
            self,
            num_classes: int,  # includes background
            CONF_THRESHOLD: float = 0.3,
            IOU_THRESHOLD: float = 0.5,
            img_idx: Optional[int] = None,
        ):
        self.confusion = np.zeros((num_classes, num_classes))
        self.confusion_with_img_indices = [
            [Counter() for _ in range(num_classes)] for _ in range(num_classes)
        ]
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
        self.img_idx = img_idx

    @abstractmethod
    def process_batch(self, detections, labels: np.ndarray, **kwargs) -> None:
        raise NotImplementedError
    
    def get_confusion(self) -> np.ndarray:
        return self.confusion

    def get_confusion_with_img_indices(self) -> List[List[Counter[int, int]]]:
        return self.confusion_with_img_indices


class DetectionConfusionMatrix(BaseConfusionMatrix):
    def process_batch(self, detections: np.array, labels: np.ndarray) -> None:
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i in range(len(labels)):
                gt_class = gt_classes[i]
                self.confusion[gt_class, 0] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i in range(len(labels)):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.confusion[gt_class, detection_class] += 1
                if self.img_idx is not None:
                    self.confusion_with_img_indices[gt_class][detection_class][self.img_idx] += 1  #
            else:
                self.confusion[gt_class, 0] += 1
                if self.img_idx is not None:
                    self.confusion_with_img_indices[gt_class][0][self.img_idx] += 1  #

        for i in range(len(detections)):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.confusion[0, detection_class] += 1
                if self.img_idx is not None:
                    self.confusion_with_img_indices[0][detection_class][self.img_idx] += 1  #


class SegmentationConfusionMatrix(BaseConfusionMatrix):
    def __init__(
            self,
            num_classes: int,  # includes background
            CONF_THRESHOLD: float = 0.3,
            IOU_THRESHOLD: float = 0.5,
            SEG_IOU_THRESHOLD: float = 0.3,
            img_idx: Optional[int] = None,
        ):
        super().__init__(num_classes, CONF_THRESHOLD, IOU_THRESHOLD, img_idx)
        self.SEG_IOU_THRESHOLD = SEG_IOU_THRESHOLD

    def process_batch(
            self,
            detections: np.array,
            labels: np.ndarray,
            detection_mask: np.array,
            label_mask: np.array
        ) -> None:
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections: (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels: (Array[M, 5]), class, x1, y1, x2, y2
            detection_mask: (Array[num_classes, H, W]), float
            label_mask: (Array[num_classes, H, W]), int
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i in range(len(labels)):
                gt_class = gt_classes[i]
                self.confusion[gt_class, 0] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i in range(len(labels)):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]

                # difference block between segmentation and detection
                cid, xmin, ymin, xmax, ymax = labels[i].tolist()
                seg_iou = self.get_seg_iou(
                    cid, xmin, ymin, xmax, ymax, label_mask, detection_mask
                )
                if seg_iou >= self.SEG_IOU_THRESHOLD:
                    self.confusion[gt_class, detection_class] += 1
                    if self.img_idx is not None:
                        self.confusion_with_img_indices[gt_class][detection_class][self.img_idx] += 1  #
                else:  # treat as false positive
                    self.confusion[gt_class, 0] += 1
                    if self.img_idx is not None:
                        self.confusion_with_img_indices[gt_class][0][self.img_idx] += 1

            else:
                self.confusion[gt_class, 0] += 1
                if self.img_idx is not None:
                    self.confusion_with_img_indices[gt_class][0][self.img_idx] += 1  #

        for i in range(len(detections)):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.confusion[0, detection_class] += 1
                if self.img_idx is not None:
                    self.confusion_with_img_indices[0][detection_class][self.img_idx] += 1  #

    def get_seg_iou(self,
            cid: int,
            xmin: int,
            ymin: int,
            xmax: int,
            ymax: int,
            label_mask: np.ndarray,
            detection_mask: np.ndarray
        ) -> float:
        """
        Args:
            cid (int): class ID
            xmin (int): x-coordinate of the top-left corner
            ymin (int): y-coordinate of the top-left corner
            xmax (int): x-coordinate of the bottom-right corner
            ymax (int): y-coordinate of the bottom-right corner
            label_mask (np.ndarray): shape=(num_of_classes, H, W) and values are int in {0, 1}
            detection_mask (np.ndarray): shape=(num_of_classes, H, W) and values are float in [0, 1]
        Return:
            iou (float)
        """
        label_mask = label_mask.copy()[cid, ymin:ymax, xmin:xmax]

        detection_mask = detection_mask.copy()[:, ymin:ymax, xmin:xmax]
        detection_mask = detection_mask.argmax(axis=0)
        detection_mask = (detection_mask == cid).astype(np.uint8)

        iou = np.sum(np.logical_and(label_mask, detection_mask)) / np.sum(np.logical_or(label_mask, detection_mask))
        return float(iou)
