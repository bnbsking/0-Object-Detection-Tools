### DETRegNotes
+ Notes after implementing https://github.com/amirbar/detreg

### YoloV4Notes
+ Notes after implementing https://github.com/WongKinYiu/PyTorch_YOLOv4

### format conversion
+ Support format conversion between Voc, Yolo and Coco
+ Support array2file auto(pseudo)-labeling for Voc and Yolo

### confusion_matrix.py
+ Compute confusion matrix from specified format

### map.py
+ Compute mAP from specified format
+ Single image mode and Total images mode

### preprocess.py
+ For yoloFloat data format only
+ Distil intersected data
+ Image shape check, class check, out-of-boundary check
+ Generate cleaned data
+ Generate data format for map.py and confusion matrix.py

### prt_ranking.ipynb
+ Plot PR-curve, P-curve, R-curve and compute best thershold from output of map.py
+ Execute map.py image-by-image. Rank the worst detected image

### visualization.py
+ support IOU and NMS processing
+ plot ground truth and prediction result among various format

### More concepts
+ object detection format overview

| - | VOC | YOLO | COCO |
| - | - | - | - |
| file | .xml per img | .txt per img | .json for all img (D['images'].id==D['annotations'].image_id) |
| format | xmin,ymin,xmax,ymax | cx,cy,w,h | xmin,ymin,w,h |
| type | int | float | int |
| classes | - | classes.txt | D['categories'] |
| raw shape | height,width | - | D['images'] |

+ IOU
+ precision and recall:
  + TP: conf>=confThreshold and IOU>=iouThreshold
  + TN: repeat detect or conf<confThreshold and IOU>=iouThreshold
  + FN: IOU<iouThreshold
+ AP = area under "smooth" PR-curve e.g. 1 if precision=recall=1
+ f = 2/(1/precision+1/recall) if precision*recall!=0 else 0. Pick best threshold from largest f if precision is as valued as recall
+ mAP = average AP among categories
  + mAP50: (default) iouThreshold=0.5
  + mAP75: iouThreshold=0.75
