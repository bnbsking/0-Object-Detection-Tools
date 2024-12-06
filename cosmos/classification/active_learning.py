import json
from math import log
import os
from typing import List, Union


class ClassificationActiveLearning:
    def __init__(self, pred_path: str, save_path: str, loss_name: str = "entropy"):
        pred = json.load(open(pred_path, 'r'))
        loss_func = getattr(self, loss_name)

        for data_dict in pred["data"]:
            data_dict["loss"] = loss_func(data_dict["pd_probs"])

        pred["data"] = sorted(pred["data"], key=lambda x: x["loss"], reverse=True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(pred, open(save_path, 'w'), indent=4)
    
    def entropy(self, pd_probs: Union[List[float], List[List[float]]]) -> float:
        if isinstance(pd_probs[0], float):
            return -sum([p * log(p + 1e-10) for p in pd_probs])
        else:
            return sum([self.entropy(p) for p in pd_probs]) / len(pd_probs)
