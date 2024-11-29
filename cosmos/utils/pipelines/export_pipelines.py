from collections import Counter
import os
import shutil
from typing import Dict, List, Optional, Set, Tuple


class ExportDataPipeline:
    def __init__(self, data_path_list: List[str], func_dicts: List[Dict], save_folder: str):
        self.data_path_list = data_path_list
        self.func_dicts = func_dicts
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

    def run(self):
        for func_dict in self.func_dicts:
            getattr(self, func_dict["func_name"])(**func_dict["func_args"])

    def _get_sorted_acc_indices(
            self,
            confusion_with_img_indices: List[List[Counter[int, int]]]
        ) -> List[Tuple[float, int]]:
        right_cnt = [0] * len(self.data_path_list)
        wrong_cnt = [0] * len(self.data_path_list)
        for i in range(len(confusion_with_img_indices)):
            for j in range(len(confusion_with_img_indices)):
                for idx, times in confusion_with_img_indices[i][j].items():
                    if i==j:
                        right_cnt[idx] += times
                    else:
                        wrong_cnt[idx] += times
        acc_indices = [(round(rc / (rc + wc + 1e-10), 3), i) \
            for i, (rc, wc) in enumerate(zip(right_cnt, wrong_cnt))]
        return sorted(acc_indices)

    def export_worst_acc_from_confusion(
            self,
            confusion_with_img_indices: List[List[Counter[int, int]]],
            top_n: Optional[int] = None
        ):
        save_folder = os.path.join(self.save_folder, "worst_acc")
        os.makedirs(save_folder, exist_ok=True)

        acc_indices = self._get_sorted_acc_indices(confusion_with_img_indices)[:top_n]
        for acc, idx in acc_indices:
            filename = os.path.basename(self.data_path_list[idx])
            save_path = os.path.join(save_folder, f"{acc}_{filename}")
            shutil.copy(self.data_path_list[idx], save_path)

    def _get_all_wrong_indices(
            self,
            confusion_with_img_indices: List[List[Counter[int, int]]]
        ) -> Set[int]:
        wrong_indices = set()
        for i in range(len(confusion_with_img_indices)):
            for j in range(len(confusion_with_img_indices)):
                if i!=j:
                    for idx in confusion_with_img_indices[i][j].keys():
                        wrong_indices.add(idx)
        return wrong_indices

    def export_all_wrong_from_confusion(
            self,
            confusion_with_img_indices: List[List[Counter[int, int]]],
        ):
        save_folder = os.path.join(self.save_folder, "all_wrong")
        os.makedirs(save_folder, exist_ok=True)

        wrong_indices = self._get_all_wrong_indices(confusion_with_img_indices)
        for idx in wrong_indices:
            filename = os.path.basename(self.data_path_list[idx])
            save_path = os.path.join(save_folder, filename)
            shutil.copy(self.data_path_list[idx], save_path)
