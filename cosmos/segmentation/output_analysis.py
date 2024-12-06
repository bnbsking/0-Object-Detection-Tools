import os
from typing import Dict, List, Literal, Optional

import numpy as np
import yaml

from ..utils.analysis.base_analysis import BaseAnalysis


class SegmentationAnalysis(BaseAnalysis):
    def __init__(
            self,
            ant_path: str,
            save_folder: str,
            task: Literal["instance", "semantic"],
            pipeline_cfg_path: Optional[str] = None
        ):
        self.task = task
        super().__init__(ant_path, save_folder, pipeline_cfg_path)

    def get_labels(self, data_dict_list: List[Dict]) -> List[np.ndarray]:
        """
        Args:
            data_dict_list (List[Dict]): A list of dicts, where each dictionary contains:
                - gt_cls (List[int]): A list of class IDs.
                - gt_boxes (List[List[int]]): A list of bounding box coordinates (xmin, ymin, xmax, ymax).
                - gt_filled_path (str): Path to the segmentation mask.
        Returns:
            labels (List[Dict]): length is num of images. Each dictionary contains:
                - detection (np.array): shape=(labels_for_an_img, 5).
                - segmentation_path (str): path to the segmentation mask.
        """
        labels = []
        for data_dict in data_dict_list:
            img_label = []
            if self.task == "instance":
                for cid, (xmin, ymin, xmax, ymax) in zip(data_dict["gt_cls"], data_dict["gt_boxes"]):
                    img_label.append([cid, xmin, ymin, xmax, ymax])
            labels.append(
                {
                    "detection": np.array(img_label),
                    "segmentation_path": data_dict["gt_filled_path"]
                }
            )
        return labels

    def get_predictions(self, data_dict_list: List[Dict]) -> List[np.ndarray]:
        """
        Args:
            data_dict_list (List[Dict]): A list of dictionaries where each dictionary contains:
                - pd_probs (List[List[float]]): shape=(num_boxes, num_classes).
                - pd_boxes (List[Tuple[int, int, int, int]]): shape=(num_boxes, 4).
                    Each box is represented as (xmin, ymin, xmax, ymax).
                - pd_filled_path (str): Path to the segmentation mask.
        Returns:
            predictions (List[Dict]): length is num of images. Each dictionary contains:
                - detection (np.array): shape=(num_boxes, 6).
                    Each prediction is represented as (xmin, ymin, xmax, ymax, conf, cid), where:
                    - xmin, ymin, xmax, ymax: Bounding box coordinates.
                    - conf: Confidence score of the prediction.
                    - cid: Class ID with the highest confidence score.
                - segmentation_path (str): path to the segmentation mask.
        """
        predictions = []
        for data_dict in data_dict_list:
            img_detect = []
            if self.task == "instance":
                for probs, (xmin, ymin, xmax, ymax) in zip(data_dict["pd_probs"], data_dict["pd_boxes"]):
                    conf = max(probs)
                    cid = probs.index(conf)
                    img_detect.append([xmin, ymin, xmax, ymax, conf, cid])
            predictions.append(
                {
                    "detection": np.array(img_detect),
                    "segmentation_path": data_dict["pd_filled_path"]
                }
            )
        return predictions
    
    def get_data_path_list(self, data_dict_list: List[Dict]) -> List[str]:
        return [data_dict["img_path"] for data_dict in data_dict_list]

    def get_pipeline_cfg(self, pipeline_cfg_path: Optional[str] = None) -> Dict:
        if pipeline_cfg_path is None:
            pipeline_cfg_path = os.path.join(\
                os.path.dirname(os.path.abspath(__file__)), f"output_analysis_{self.task}.yaml"
            )
        return yaml.safe_load(open(pipeline_cfg_path, "r"))
