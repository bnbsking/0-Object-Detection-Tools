import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class PlottingPipeline:
    def __init__(self, class_list: List[str], func_dicts: List[Dict], save_folder: str):
        self.class_list = class_list
        self.func_dicts = func_dicts
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

    def run(self):
        for func_dict in self.func_dicts:
            getattr(self, func_dict["func_name"])(**func_dict["func_args"])
    
    def plot_aps(self, ap_list: List[float], map: float, wmap: float):
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.set_title(f"map={round(map, 3)}, wmap={round(wmap, 3)}", fontsize=16)
        ax.bar(self.class_list, ap_list)
        for i in range(len(self.class_list)):
            ax.text(i, ap_list[i], ap_list[i], ha="center", va="bottom", fontsize=16)
        plt.savefig(os.path.join(self.save_folder, "aps.jpg"))
        plt.show()

    def plot_pr_curves(self, refine_pr_curves: List[Dict[str, List[float]]]):
        num_classes = len(refine_pr_curves)
        plt.figure(figsize=(6 * num_classes, 4))
        for cid in range(num_classes):
            plt.subplot(1, num_classes, 1 + cid)
            plt.scatter(refine_pr_curves[cid]["refine_recall"], refine_pr_curves[cid]["refine_precision"])
            plt.plot(refine_pr_curves[cid]["refine_recall"], refine_pr_curves[cid]["refine_precision"])
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.grid('on')
            plt.title(f"{cid}-{self.class_list[cid]}", fontsize=16)
            plt.xlabel("recall", fontsize=16)
            plt.ylabel("precision", fontsize=16)
        plt.savefig(os.path.join(self.save_folder, "pr_curves.jpg"))
        plt.show()

    def plot_prf_curves(self, pr_curves: List[Dict[str, List[float]]]):
        num_classes = len(pr_curves)
        plt.figure(figsize=(6 * num_classes, 4))
        for cid in range(num_classes):
            f1_arr = [2 * p * r / (p + r + 1e-10) for p, r in \
                zip(pr_curves[cid]["precision"], pr_curves[cid]["recall"])]
            plt.subplot(1, num_classes, 1 + cid)
            plt.plot(pr_curves[cid]["precision"])
            plt.plot(pr_curves[cid]["recall"])
            plt.plot(f1_arr)
            plt.xlim(-5, 105)
            plt.ylim(-0.05, 1.05)
            plt.grid('on')
            plt.title(f"{cid}-{self.class_list[cid]}", fontsize=16)
            plt.xlabel("threshold", fontsize=16)
            plt.legend(labels=["precision", "recall", "f1"], fontsize=12)
        plt.savefig(os.path.join(self.save_folder, "prf_curves.jpg"))
        plt.show()

    def plot_confusion(
            self,
            confusion: np.ndarray,
            confusion_col_norm: np.ndarray,
            confusion_row_norm: np.ndarray
        ):
        num_classes = len(self.class_list)
        matrix_plot_list = [confusion_col_norm, confusion_col_norm, confusion_row_norm]
        matrix_text_list = [confusion, confusion_col_norm, confusion_row_norm]
        title_list = ["confusion", "col norm (precision)", "row norm (recall)"]

        plt.figure(figsize=(15,5))
        
        for i, (matplt, mattxt, title) in enumerate(zip(matrix_plot_list, matrix_text_list, title_list)):
            fig = plt.subplot(1, 3, 1+i)
            plt.title(title, fontsize=12)
            plt.xlabel("PD", fontsize=12)
            plt.ylabel("GT", fontsize=12)
            fig.set_xticks(np.arange(num_classes + 1)) # values
            fig.set_xticklabels(self.class_list + ['BG'])  # labels
            fig.set_yticks(np.arange(num_classes + 1))  # values
            fig.set_yticklabels(self.class_list + ['BG'])  # labels
            plt.imshow(matplt, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
            for i in range(num_classes + 1):
                for j in range(num_classes + 1):
                    plt.text(j, i, round(mattxt[i][j], 2), ha="center", va="center", \
                        color="black" if matplt[i][j]<0.9 else "white", fontsize=12)
        plt.savefig(os.path.join(self.save_folder, "confusion.jpg"))
        plt.show()