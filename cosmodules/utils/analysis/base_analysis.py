from abc import abstractmethod
import json
import os
from typing import Dict, List, Optional

import numpy as np
import yaml

from .. import pipelines


class BaseAnalysis:
    def __init__(
            self,
            ant_path: str,
            save_folder: str,
            pipeline_cfg_path: Optional[str] = None
        ):
        # initialization
        general = json.load(open(ant_path, "r", encoding="utf-8"))
        class_list = general["categories"]
        start_idx = int(class_list[0] == "__background__")
        num_classes = len(general["categories"])
        labels = self.get_labels(general["data"])
        predictions = self.get_predictions(general["data"])
        data_path_list = self.get_data_path_list(general["data"])
        pipeline_cfg = self.get_pipeline_cfg(pipeline_cfg_path)

        # metrics pipeline
        metrics_pipeline_cls = getattr(pipelines, pipeline_cfg["metrics_pipeline"]["name"])
        metrics_pipeline = metrics_pipeline_cls(
            num_classes = num_classes,
            labels = labels,
            predictions = predictions,
            func_dicts = pipeline_cfg["metrics_pipeline"]["func_dicts"],
            save_path = os.path.join(save_folder, "metrics.json"),
            start_idx = start_idx
        )
        metrics = metrics_pipeline.run()

        # plotting pipeline
        plotting_pipeline_cls = getattr(pipelines, pipeline_cfg["plotting_pipeline"]["name"])
        func_dicts = self.update_args_by_metrics(
            metrics = metrics,
            func_dicts = pipeline_cfg["plotting_pipeline"]["func_dicts"],
        )
        plotting_pipeline = plotting_pipeline_cls(
            class_list = class_list[start_idx:],
            func_dicts = func_dicts,
            save_folder = save_folder
        )
        plotting_pipeline.run()

        # export pipeline
        export_pipeline_cls = getattr(pipelines, pipeline_cfg["export_pipeline"]["name"])
        func_dicts = self.update_args_by_metrics(
            metrics = metrics,
            func_dicts = pipeline_cfg["export_pipeline"]["func_dicts"],
        )
        export_pipeline = export_pipeline_cls(
            data_path_list = data_path_list,
            func_dicts = func_dicts,
            save_folder = save_folder
        )
        export_pipeline.run()

    @abstractmethod
    def get_labels(self, data_dict_list: List[Dict]):
        raise NotImplementedError

    @abstractmethod
    def get_predictions(self, data_dict_list: List[Dict]):
        raise NotImplementedError
    
    @abstractmethod
    def get_data_path_list(self, data_dict_list: List[Dict]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_pipeline_cfg(self, pipeline_cfg_path: Optional[str] = None) -> Dict:
        raise NotImplementedError
    
    def update_args_by_metrics(self, metrics: Dict, func_dicts: Dict) -> Dict:
        for func_dict in func_dicts:
            for k, v in func_dict["func_args"].items():
                func_dict["func_args"][k] = metrics[v]
        return func_dicts
