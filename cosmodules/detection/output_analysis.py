import os
from typing import Dict, List, Optional

import numpy as np
import yaml

from ..utils.analysis.base_analysis import BaseAnalysis


class DetectionAnalysis(BaseAnalysis):
    def __init__(
            self,
            ant_path: str,
            save_folder: str,
            pipeline_cfg_path: Optional[str] = None
        ):
        super().__init__(ant_path, save_folder, pipeline_cfg_path)

    def get_labels(self, data_dict_list: List[Dict]) -> List[np.ndarray]:
        """
        Args:
            data_dict_list (List[Dict]): A list of dicts, where each dictionary contains:
                - gt_cls (List[int]): A list of class IDs.
                - gt_boxes (List[List[int]]): A list of bounding box coordinates (xmin, ymin, xmax, ymax).
        Returns:
            labels (List[np.ndarray]): length is num of images. each shape=(labels_for_an_img, 5).
                Each label is represented as (class_id, xmin, ymin, xmax, ymax).
        """
        labels = []
        for data_dict in data_dict_list:
            img_label = []
            for cid, (xmin, ymin, xmax, ymax) in zip(data_dict["gt_cls"], data_dict["gt_boxes"]):
                img_label.append([cid, xmin, ymin, xmax, ymax])
            labels.append(np.array(img_label))
        return labels

    def get_predictions(self, data_dict_list: List[Dict]) -> List[np.ndarray]:
        """
        Args:
            data_dict_list (List[Dict]): A list of dictionaries where each dictionary contains:
                - pd_probs (List[List[float]]): shape=(num_boxes, num_classes).
                - pd_boxes (List[Tuple[int, int, int, int]]): shape=(num_boxes, 4).
                    Each box is represented as (xmin, ymin, xmax, ymax).
        Returns:
            predictions (List[np.ndarray]): length is num of images. each shape=(num_boxes, 6).
                Each prediction is represented as (xmin, ymin, xmax, ymax, conf, cid), where:
                - xmin, ymin, xmax, ymax: Bounding box coordinates.
                - conf: Confidence score of the prediction.
                - cid: Class ID with the highest confidence score.
        """
        predictions = []
        for data_dict in data_dict_list:
            img_detect = []
            for probs, (xmin, ymin, xmax, ymax) in zip(data_dict["pd_probs"], data_dict["pd_boxes"]):
                conf = max(probs)
                cid = probs.index(conf)
                img_detect.append([xmin, ymin, xmax, ymax, conf, cid])
            predictions.append(np.array(img_detect))
        return predictions
    
    def get_data_path_list(self, data_dict_list: List[Dict]) -> List[str]:
        return [data_dict["img_path"] for data_dict in data_dict_list]

    def get_pipeline_cfg(self, pipeline_cfg_path: Optional[str] = None) -> Dict:
        if pipeline_cfg_path is None:
            pipeline_cfg_path = os.path.join(\
                os.path.dirname(os.path.abspath(__file__)), "output_analysis.yaml"
            )
        return yaml.safe_load(open(pipeline_cfg_path, "r"))
