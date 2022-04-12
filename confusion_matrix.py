# e.g. python confusion_matrix.py 4 0.31 ../data/labv2/testv2/yoloIntAnt/ ../exps/xavier_messy3k_DETReg_fine_tune_full_coco/txt
import numpy as np


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

# e.g. python confusion_matrix.py 4 0.31 ../data/labv2/testv2/yoloIntAnt/ ../exps/xavier_messy3k_DETReg_fine_tune_full_coco/txt
import sys, glob
import matplotlib.pyplot as plt
import matplotlib as mpl

_, classes, threshold, gtPath, dtPath = sys.argv
classL, threshold, gtPathL, dtPathL = classes.split(","), float(threshold), sorted(glob.glob(f"{gtPath}/*.txt")), sorted(glob.glob(f"{dtPath}/*.txt"))
n = len(classL)
assert len(gtPathL)==len(dtPathL)

M = np.zeros( (n+1,n+1) ) # col:gt, row:pd
for i,(gtPath,dtPath) in enumerate(zip(gtPathL,dtPathL)):
    with open(gtPath,"r") as f:
        labels = []
        for line in f.readlines():
            cid, xmin, ymin, xmax, ymax = line.replace("\n","").split(" ")
            labels.append( [int(cid), int(xmin), int(ymin), int(xmax), int(ymax)] )
        labels = np.array(labels)
    with open(dtPath,"r") as f:
        detections = []
        for line in f.readlines():
            cid, conf, xmin, ymin, xmax, ymax = line.replace("\n","").split(" ")
            detections.append( [int(xmin), int(ymin), int(xmax), int(ymax), float(conf), int(cid)] )
        detections = np.array(detections)
    cm = ConfusionMatrix(n, CONF_THRESHOLD=threshold, IOU_THRESHOLD=0.5)
    cm.process_batch(detections,labels)
    M += cm.return_matrix()
    #if cm.return_matrix()[:,4].sum()>0: # unknown error, last column should be zero
    #    print(gtPath,dtPath)
    #    print(cm.return_matrix())
    #if i>10:
    #    break
N = M / M.sum(axis=0)

plt.figure(figsize=(10,5))
#
fig = plt.subplot(1,2,1)
plt.title(f"Confusion Matrix - Number", fontsize=12)
plt.xlabel("GT", fontsize=12)
plt.ylabel("PD", fontsize=12)
fig.set_xticks(np.arange(n+1), classL+['BG'])
fig.set_yticks(np.arange(n+1), classL+['BG'])
plt.imshow(N, cmap=mpl.cm.Blues, interpolation='nearest')
for i in range(n+1):
    for j in range(n+1):
        plt.text(j, i, int(M[i][j]), ha="center", va="center", color="black" if N[i][j]<0.9 else "white", fontsize=12)
#
fig = plt.subplot(1,2,2)
plt.title(f"Confusion Matrix - Ratio", fontsize=12)
plt.xlabel("GT", fontsize=12)
plt.ylabel("PD", fontsize=12)
fig.set_xticks(np.arange(n+1), classL+['BG'])
fig.set_yticks(np.arange(n+1), classL+['BG'])
plt.imshow(N, cmap=mpl.cm.Blues, interpolation='nearest')
for i in range(n+1):
    for j in range(n+1):
        plt.text(j, i, round(N[i][j],2), ha="center", va="center", color="black" if N[i][j]<0.9 else "white", fontsize=12)
plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.cm.Blues))
#
plt.savefig("result.jpg")
