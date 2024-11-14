import json, glob, os
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import confusion_matrix
#import visualization as viz


class Result:
    """
    valTxtPath = "/home/jovyan/data-vol-1/yolov7/_data/sample200new/val.txt"
    pdJsonPath = "/home/jovyan/data-vol-1/yolov7/runs/test/sample200_lr1_raw/best_predictions.json"
    obj = Result(valTxtPath, pdJsonPath, classL=['tetra'], classNumL=[1])
    obj.getPR()
    obj.getRefineRP()
    obj.getAPs()
    obj.plotPR()
    obj.plotConfusion()
    obj.getBlockImgs(1,0)
    """
    # def __init__(self, imgPathL, antPathL, labels, detections, classL, classNumL, savePath):
    #     self.imgPathL = imgPathL
    #     self.antPathL = antPathL
    #     self.labels   = labels
    #     self.detections = detections
    #     self.classL    = classL
    #     self.classNumL = classNumL
    #     self.savePath  = savePath
    
    def __init__(self, ant_path: str, save_folder: str):
        with open(ant_path, "r", encoding="utf-8") as f:
            general = json.load(f)
        
        class_list = general["categories"]
        num_classes = len(general["categories"])
        labels = self.get_labels(general["data"])
        detections = self.get_detections(general["data"])

        pr_curves = self.get_pr_curves(num_classes, labels, detections)
        #print(pr_curves)
        pr_curves = self.get_refine_pr_curves(pr_curves)
        #print(pr_curves)
        aps = self.get_aps(pr_curves, None)
        print(aps)

        self.plot_aps(class_list, aps, save_folder)
        self.plot_pr_curves(class_list, pr_curves, save_folder)
        self.plot_prf_curves(class_list, pr_curves, save_folder)

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

    def get_pr_curves(
            self,
            num_classes: int,
            labels: List[np.ndarray],
            detections: List[np.ndarray],
            k: int = 101
        ) -> List[Dict[str, List[float]]]:
        pr_curves = [
            {
                "precision": [0.] * k,
                "recall": [0.] * k,
            } for _ in range(num_classes)
        ]
        for i, threshold in tqdm(enumerate(np.linspace(0, 1, k))):
            # get confusion of the threshold
            confusion = np.zeros((num_classes+1, num_classes+1))  # (i, j) = (pd, gt)
            for label, detection in zip(labels, detections):
                img_confusion = confusion_matrix.ConfusionMatrix(
                    num_classes,
                    CONF_THRESHOLD = threshold,
                    IOU_THRESHOLD = 0.5
                )
                img_confusion.process_batch(detection, label)
                confusion += img_confusion.return_matrix()
            
            # update pr curve at the threshold from confusion
            row_sum = confusion.sum(axis=1)
            col_sum = confusion.sum(axis=0)
            for cid in range(num_classes):
                pr_curves[cid]["precision"][i] = confusion[cid][cid]/row_sum[cid] if row_sum[cid] else 0
                pr_curves[cid]["recall"][i] = confusion[cid][cid]/col_sum[cid] if col_sum[cid] else 0
        return pr_curves

    def get_refine_pr_curves(
            self,
            pr_curves: List[Dict[str, List[float]]]
        ) -> List[Dict[str, List[float]]]:
        """
        sorted by recall, and enhance precision by next element reversely
        """
        for cid in range(len(pr_curves)):
            recall_arr = pr_curves[cid]["recall"].copy()
            precision_arr = pr_curves[cid]["precision"].copy()
            zip_arr = sorted(zip(recall_arr, precision_arr))
            recall_arr, precision_arr = zip(*zip_arr)
            recall_arr, precision_arr = list(recall_arr), list(precision_arr)
            for i in range(1, len(precision_arr)):
                precision_arr[-1-i] = max(precision_arr[-1-i], precision_arr[-i])
            pr_curves[cid]["refine_recall"] = recall_arr
            pr_curves[cid]["refine_precision"] = precision_arr
        return pr_curves
    
    def get_aps(
            self,
            pr_curves: List[Dict[str, List[float]]],
            gt_class_cnts: List[int]
        ) -> Dict:
        aps = {
            "ap_list": [],
            "map": -1,
            "wmap": -1
        }
        num_classes = len(pr_curves)
        for cid in range(num_classes):
            ap = 0
            for i in range(len(pr_curves[0]["precision"])-1):
                ap += pr_curves[cid]["refine_precision"][i] * \
                    (pr_curves[cid]["refine_recall"][i+1] - pr_curves[cid]["refine_recall"][i])
            aps["ap_list"].append(round(ap,3))
        aps["map"] = sum(aps["ap_list"]) / num_classes
        aps["wmap"] = sum(ap * cnt for ap, cnt in zip(aps["ap_list"], gt_class_cnts))\
            / sum(gt_class_cnts) if gt_class_cnts else -1
        return aps
    
    def plot_aps(self, class_list: List[str], aps: Dict, save_folder: str):
        os.makedirs(save_folder, exist_ok=True)
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.set_title(f"map={aps.get('map', -1)}, wmap={aps.get('wmap', -1)}", fontsize=16)
        ax.bar(class_list, aps["ap_list"])
        for i in range(len(class_list)):
            ax.text(i, aps["ap_list"][i], aps["ap_list"][i], ha="center", va="bottom", fontsize=16)
        plt.savefig(os.path.join(save_folder, "aps.jpg"))
        plt.show()

    def plot_pr_curves(self, class_list: List[str], pr_curves: List[Dict[str, List[float]]], save_folder: str):
        os.makedirs(save_folder, exist_ok=True)
        num_classes = len(pr_curves)
        plt.figure(figsize=(6 * num_classes, 4))
        for cid in range(num_classes):
            plt.subplot(1, num_classes, 1 + cid)
            plt.scatter(pr_curves[cid]["refine_recall"], pr_curves[cid]["refine_precision"])
            plt.plot(pr_curves[cid]["refine_recall"], pr_curves[cid]["refine_precision"])
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.grid('on')
            plt.title(f"{cid}-{class_list[cid]}", fontsize=16)
            plt.xlabel("recall", fontsize=16)
            plt.ylabel("precision", fontsize=16)
        plt.savefig(os.path.join(save_folder, "pr_curves.jpg"))
        plt.show()

    def plot_prf_curves(self, class_list: List[str], pr_curves: List[Dict[str, List[float]]], save_folder: str):
        os.makedirs(save_folder, exist_ok=True)
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
            plt.title(f"{cid}-{class_list[cid]}", fontsize=16)
            plt.xlabel("threshold", fontsize=16)
            plt.legend(labels=["precision", "recall", "f1"], fontsize=12)
        plt.savefig(os.path.join(save_folder, "prf_curves.jpg"))
        plt.show()

    

    def _getBestThreshold(self):
        n = len(self.classL)
        wScore = [0]*101
        for i in range(101):
            for j in range(n):                
                p, r = self.PR[j]["precision"][i], self.PR[j]["recall"][i]
                if self.strategy=="fvalue":
                    score = 2*p*r/(p+r) if p+r else 0
                elif self.strategy=="precision":
                    score = p if r>=0.5 else 0
                else:
                    raise
                wScore[i] += score*self.classNumL[j]/sum(self.classNumL)
        bestScore, self.bestThreshold = max(zip(wScore,[round(0.01*i,2) for i in range(101)]))
        print(f"bestScore={round(bestScore,2)}, best_threshold={self.bestThreshold}")
    
    def plotConfusion(self, strategy="fvalue"):
        self.strategy = strategy
        self._getBestThreshold()
        n = len(self.classL)
        M = np.zeros( (n+1,n+1) ) # col:gt, row:pd
        self.accumFileL = [ [[] for j in range(n+1)] for i in range(n+1) ] # (n+1,n+1) each grid is path list
        for j,(imgPath,labels,detections) in enumerate(zip(self.imgPathL,self.labels,self.detections)):
            cm = confusion_matrix.ConfusionMatrix(n, CONF_THRESHOLD=self.bestThreshold, IOU_THRESHOLD=0.5, gtFile=imgPath, accumFileL=self.accumFileL)
            cm.process_batch(detections,labels)
            M += cm.return_matrix()
            self.accumFileL = cm.getAccumFileL()
        #
        #print(M)
        axis0sum = M.sum(axis=0)
        N = M.copy()
        for i in range(len(N)):
            if axis0sum[i] != 0:
                N[:,i] /= axis0sum[i]
        #print(N)
        axis1sum = M.sum(axis=1)
        P = M.copy()
        for i in range(len(P)):
            if axis1sum[i] != 0:
                P[i,:] /= axis1sum[i]
        #print(P)
        #
        plt.figure(figsize=(15,5))
        # fig1 - number
        fig = plt.subplot(1,3,1)
        plt.title(f"Confusion Matrix - Number (conf={self.bestThreshold})", fontsize=12)
        plt.xlabel("GT", fontsize=12)
        plt.ylabel("PD", fontsize=12)
        fig.set_xticks(np.arange(n+1)) # values
        fig.set_xticklabels(self.classL+['BG']) # labels
        fig.set_yticks(np.arange(n+1)) # values
        fig.set_yticklabels(self.classL+['BG']) # labels
        plt.imshow(P, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
        for i in range(n+1):
            for j in range(n+1):
                plt.text(j, i, int(M[i][j]), ha="center", va="center", color="black" if P[i][j]<0.9 else "white", fontsize=12)
        # fig2 - precision
        fig = plt.subplot(1,3,2)
        plt.title(f"Confusion Matrix - Row norm (Precision)", fontsize=12)
        plt.xlabel("GT", fontsize=12)
        plt.ylabel("PD", fontsize=12)
        fig.set_xticks(np.arange(n+1)) # values
        fig.set_xticklabels(self.classL+['BG']) # labels
        fig.set_yticks(np.arange(n+1)) # values
        fig.set_yticklabels(self.classL+['BG']) # labels
        plt.imshow(P, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
        for i in range(n+1):
            for j in range(n+1):
                plt.text(j, i, round(P[i][j],2), ha="center", va="center", color="black" if P[i][j]<0.9 else "white", fontsize=12)
        # fig3 - recall
        fig = plt.subplot(1,3,3)
        plt.title(f"Confusion Matrix - Col norm (Recall)", fontsize=12)
        plt.xlabel("GT", fontsize=12)
        plt.ylabel("PD", fontsize=12)
        fig.set_xticks(np.arange(n+1)) # values
        fig.set_xticklabels(self.classL+['BG']) # labels
        fig.set_yticks(np.arange(n+1)) # values
        fig.set_yticklabels(self.classL+['BG']) # labels
        plt.imshow(N, cmap=mpl.cm.Blues, interpolation='nearest', vmin=0, vmax=1)
        for i in range(n+1):
            for j in range(n+1):
                plt.text(j, i, round(N[i][j],2), ha="center", va="center", color="black" if N[i][j]<0.9 else "white", fontsize=12)
        #plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.cm.Blues))
        plt.savefig(f"{self.savePath}/confusion.jpg")
        plt.show()
        json.dump(self.accumFileL, open(f"{self.savePath}/confusionFiles.json","w"))
        
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

obj = Result(
    ant_path = "../example/data/gt_and_pd.json",
    save_folder = "."
)
