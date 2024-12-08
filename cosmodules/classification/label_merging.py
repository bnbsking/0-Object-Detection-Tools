from collections import Counter
import copy
import json
import os
import random
from typing import Dict, List, Literal, Optional, Tuple


class ClassificationLabelMerging:
    def __init__(
            self,
            cfg_path_list: List[str],
            save_path: str,
            ties_handling: Literal["random", "null"] = "null"
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
                assert os.path.basename(data_dict["data_path"]) \
                    == os.path.basename(data_dict0["data_path"])
                assert type(data_dict["gt_cls"]) == type(data_dict0["gt_cls"])

    def merge_gt_cls(
            self,
            collect_gt_cls: List[int],
            ties_handling: Literal["random", "null"] = "null"
        ) -> Tuple[bool, Optional[int]]:
        votes = Counter(collect_gt_cls)
        max_vote = max(votes.values())
        max_vote_cls = [k for k, v in votes.items() if v == max_vote]
        
        if len(max_vote_cls) == 1:
            return True, max_vote_cls[0]
        elif ties_handling == "random":
            return False, random.choice(max_vote_cls)
        else:
            return False, None

    def merge(
            self,
            cfg_list: List[Dict],
            ties_handling: Literal["random", "null"] = "null"
        ) -> Dict:
        cfg_merged = copy.deepcopy(cfg_list[0])
        if isinstance(cfg_list[0]["data"][0]["gt_cls"], int):
            label_dim = 0
        else:
            label_dim = len(cfg_list[0]["data"][0]["gt_cls"])

        for i in range(len(cfg_list[0]["data"])):
            if label_dim == 0:
                collect_gt_cls = [cfg["data"][i]["gt_cls"] for cfg in cfg_list]
                is_consensus, merged_cls = self.merge_gt_cls(collect_gt_cls, ties_handling)
                cfg_merged["data"][i]["gt_cls"] = merged_cls
                cfg_merged["data"][i]["controversial"] = not is_consensus
                
            else:
                cfg_merged["data"][i]["gt_cls"] = [None] * label_dim
                cfg_merged["data"][i]["controversial"] = [False] * label_dim

                for j in range(label_dim):
                    collect_gt_cls = [cfg["data"][i]["gt_cls"][j] for cfg in cfg_list]
                    is_consensus, merged_cls = self.merge_gt_cls(collect_gt_cls, ties_handling)
                    cfg_merged["data"][i]["gt_cls"][j] = merged_cls
                    cfg_merged["data"][i]["controversial"][j] = not is_consensus
        
        return cfg_merged