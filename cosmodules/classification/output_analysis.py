import os
from typing import Dict, List, Optional

import numpy as np
import yaml

from ..utils.analysis.base_analysis import BaseAnalysis


class ClassificationAnalysis(BaseAnalysis):
    def __init__(
            self,
            ant_path: str,
            save_folder: str,
            pipeline_cfg_path: Optional[str] = None
        ):
        super().__init__(ant_path, save_folder, pipeline_cfg_path)

    def get_labels(self, data_dict_list: List[Dict]) -> np.array:
        """
        Returns:
            labels (np.ndarray): shape=(data,) or shape=(data, multi-label-dim)
        """
        return np.array([data_dict["gt_cls"] for data_dict in data_dict_list])

    def get_predictions(self, data_dict_list: List[Dict]) -> np.ndarray:
        """
        Returns:
            prediction (np.ndarray): shape=(data, num_classes) or (data, multi-label-dim, 2)
        """
        return np.array([data_dict["pd_probs"] for data_dict in data_dict_list])
    
    def get_data_path_list(self, data_dict_list: List[Dict]) -> List[str]:
        return [data_dict["data_path"] for data_dict in data_dict_list]

    def get_pipeline_cfg(self, pipeline_cfg_path: Optional[str] = None) -> Dict:
        if pipeline_cfg_path is None:
            pipeline_cfg_path = os.path.join(\
                os.path.dirname(os.path.abspath(__file__)), "output_analysis.yaml"
            )
        return yaml.safe_load(open(pipeline_cfg_path, "r"))
