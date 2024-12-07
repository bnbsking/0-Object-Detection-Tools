from collections import Counter
import copy
import json
import os
import random
from typing import Dict, List, Literal, Optional, Tuple

from ..utils.detection.tools import get_iou


class DetectionLabelMerging:
    def __init__(
            self,
            cfg_path_list: List[str],
            save_path: str,
            ties_handling: Literal["union", "drop"] = "union"
        ):
        cfg_list = []
        for cfg_path in cfg_path_list:
            with open(cfg_path, 'r') as f:
                cfg_list.append(json.load(f))
        self.format_consistency_check(cfg_list)
        cfg_merged = self.merge(cfg_list, ties_handling)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(cfg_merged, f, indent=4)

    def format_consistency_check(self, cfg_list: List[Dict]):
        cfg0 = cfg_list[0]
        for cfg in cfg_list[1:]:
            assert cfg["categories"] == cfg0["categories"]
            assert len(cfg["data"]) == len(cfg0["data"])
            for data_dict, data_dict0 in zip(cfg["data"], cfg0["data"]):
                assert os.path.basename(data_dict["img_path"]) \
                    == os.path.basename(data_dict0["img_path"])
                assert type(data_dict["gt_cls"]) == type(data_dict0["gt_cls"])

    def merge(
            self,
            cfg_list: List[Dict],
            ties_handling: Literal["union", "drop"] = "union",
            iou_threshold: float = 0.5
        ) -> Dict:
        cfg_merged = copy.deepcopy(cfg_list[0])

        # iterate images
        for i in range(len(cfg_list[0]["data"])):
            cfg_merged["data"][i]["gt_boxes"] = []
            cfg_merged["data"][i]["gt_cls"] = []
            cfg_merged["data"][i]["controversial"] = []

            # for each labeler, shape = (labelers, len_boxes, 3),
            # where the 3 is (score: int, box: List[int], cls: int)
            labels = []
            for cfg in cfg_list:
                labels.append(
                    [
                        [0, box, cls] \
                            for box, cls in zip(cfg["data"][i]["gt_boxes"], cfg["data"][i]["gt_cls"])
                    ]
                )
            
            # iterate all labelers
            while labels:
                # iterate boxes and cls of the last labeler
                for _, (xmin1, ymin1, xmax1, ymax1), cls1 in labels[-1]:
                    matches = 1
                    new_box = [xmin1, ymin1, xmax1, ymax1]

                    # iterate other labelers
                    for j in range(len(labels) - 1):
                        # iterate boxes and cls of other labeler
                        for k in range(len(labels[j])):
                            _, (xmin2, ymin2, xmax2, ymax2), cls2 = labels[j][k]
                            iou = get_iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                            score = iou if cls1 == cls2 and iou >= iou_threshold else 0
                            labels[j][k][0] = score
                        
                        labels[j].sort()  # sort by (score, box, cls) of other-labeler-j
                        _, (xmin, ymin, xmax, ymax), _ = labels[j][-1]  # best match of other-labeler-j
                        if labels[j][-1][0] > 0:  # other-labeler-j 's best match score > 0
                            matches += 1
                            new_box[0] += xmin
                            new_box[1] += ymin
                            new_box[2] += xmax
                            new_box[3] += ymax
                            labels[j].pop()  # remove the best match
                    
                    controversial = matches <= len(cfg_list) / 2
                    if controversial or ties_handling == "union":
                        new_box = [int(x / matches) for x in new_box]
                        cfg_merged["data"][i]["gt_boxes"].append(new_box)
                        cfg_merged["data"][i]["gt_cls"].append(cls1)
                        cfg_merged["data"][i]["controversial"].append(controversial)
                
                labels.pop()
        
        return cfg_merged