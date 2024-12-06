from collections import Counter
import json
import os
from typing import Dict, List, Union


class ClassificationLabelMerging:
    def __init__(self, cfg_path_list: List[str], save_path: str):
        cfg_list = []
        for cfg_path in cfg_path_list:
            with open(cfg_path, 'r') as f:
                cfg_list.append(json.load(f))
        self.format_consistency_check(cfg_list)
        self.merge(cfg_list, save_path)

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
            collect_gt_cls: Union[List[int], List[List[int]]]
        ) -> Union[int, List[int], None]:
        if isinstance(collect_gt_cls[0], int):
            votes = Counter(collect_gt_cls)
        else:
            votes = Counter(map(tuple, collect_gt_cls))

        max_vote = max(votes.values())
        max_vote_cls = [k for k, v in votes.items() if v == max_vote]
        if len(max_vote_cls) == 1:
            consensus = max_vote_cls[0]
            return consensus if isinstance(consensus, int) else list(consensus)
        else:
            return None

    def merge(self, cfg_list: List[Dict], save_path: str):
        cfg_merged = cfg_list[0].copy()
        cfg_merged["controversial_indices"] = []
        
        for i in range(len(cfg_merged["data"])):
            collect_gt_cls = []
            for cfg in cfg_list:
                collect_gt_cls.append(cfg["data"][i]["gt_cls"])

            merged = self.merge_gt_cls(collect_gt_cls)
            if merged is not None:
                cfg_merged["data"][i]["gt_cls"] = merged
            else:
                cfg_merged["data"][i]["gt_cls"] = None
                cfg_merged["controversial_indices"].append(i)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(cfg_merged, f, indent=4)