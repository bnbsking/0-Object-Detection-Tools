import json
import glob
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from ..pipelines import DetectionMetricsPipeline
from ..pipelines import PlottingPipeline


class DetectionAnalysis:
    def __init__(
            self,
            ant_path: str,
            save_folder: str,
            cfg_path: Optional[str] = None
        ):
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
        
        class_list = general["categories"]
        num_classes = len(general["categories"])
        labels = self.get_labels(general["data"])
        detections = self.get_detections(general["data"])
        img_path_list = self.get_img_path_list(general["data"])
        pipelines_func_dicts = self.get_pipelines_func_dicts(cfg_path)

        metrics_pipeline = DetectionMetricsPipeline(
            num_classes = num_classes,
            labels = labels,
            detections = detections,
            func_dicts = pipelines_func_dicts["metrics_pipeline_func_dicts"]
        )
        metrics = metrics_pipeline.run()
        self.save_json(metrics, os.path.join(save_folder, "metrics.json"))
        print(metrics)

        func_dicts = self.update_args_by_metrics(
            metrics = metrics,
            func_dicts = pipelines_func_dicts["plotting_pipeline_func_dicts"],
        )
        plotting = PlottingPipeline(
            class_list = class_list,
            save_folder = save_folder,
            func_dicts = func_dicts
        )
        plotting.run()

    def get_labels(self, data_dict_list: List[Dict]) -> List[np.ndarray]:
        labels = []
        for data_dict in data_dict_list:
            img_label = []
            for cid, (xmin, ymin, xmax, ymax) in zip(data_dict["gt_cls"], data_dict["gt_boxes"]):
                img_label.append([cid, xmin, ymin, xmax, ymax])
            labels.append(np.array(img_label))
        return labels

    def get_detections(self, data_dict_list: List[Dict]) -> List[np.ndarray]:
        detections = []
        for data_dict in data_dict_list:
            img_detect = []
            for probs, (xmin, ymin, xmax, ymax) in zip(data_dict["pd_probs"], data_dict["pd_boxes"]):
                conf = max(probs)
                cid = probs.index(conf)
                img_detect.append([xmin, ymin, xmax, ymax, conf, cid])
            detections.append(np.array(img_detect))
        return detections
    
    def get_img_path_list(self, data_dict_list: List[Dict]) -> List[str]:
        return [data_dict["img_path"] for data_dict in data_dict_list]

    def get_pipelines_func_dicts(self, cfg_path: Optional[str] = None) -> Dict:
        if cfg_path is None:
            cfg_path = os.path.join(\
                os.path.dirname(os.path.abspath(__file__)), "output_analysis.yaml"
            )
        pipelines_func_dicts = yaml.safe_load(open(cfg_path, "r"))
        return pipelines_func_dicts

    def deserialize(self, data: Dict):
        if isinstance(data, dict):
            return {k: self.deserialize(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def save_json(self, data: Dict, save_path: str) -> None:
        with open(save_path, "w") as f:
            json.dump(self.deserialize(data), f, indent=4)
    
    def update_args_by_metrics(self, metrics: Dict, func_dicts: Dict) -> Dict:
        for func_dict in func_dicts:
            for k, v in func_dict["func_args"].items():
                func_dict["func_args"][k] = metrics[v]
        return func_dicts

    def getBlockImgs(self, row, col, threshold='best'): # PD, GT
        classL = self.classL + ['BG']
        folder = f"{self.savePath}/GT_{classL[col]}_PD_{classL[row]}"
        os.makedirs(folder, exist_ok=True)
        threshold = self.bestThreshold if threshold=='best' else threshold 
        for imgPath in self.accumFileL[row][col]:
            idx = self.imgPathL.index(imgPath)
            det = self.detections[idx]
            det = det[det[:,4]>threshold]
            viz.show(self.classL, imgPath, imgPath.replace('.jpg','.xml'), 'voc', det[:,:4], det[:,5].astype(int), det[:,4], folder )
        print(f"len(glob.glob(folder+'/*.jpg'))={len(glob.glob(folder+'/*.jpg'))}")
