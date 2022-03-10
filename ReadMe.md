#### Convert.ipynb: Data format transformation and pseudo labeling tool

#### Visualization.ipynb: Plot ground truth and prediction result among various format

#### Object detection format overview 

| - | VOC | YOLO | COCO |
| - | - | - | - |
| file | .xml per img | .txt per img | .json for all img (D['images'].id==D['annotations'].image_id) |
| format | xmin,ymin,xmax,ymax | cx,cy,w,h | xmin,ymin,w,h |
| type | int | float | int |
| classes | - | classes.txt | D['categories'] |
| raw shape | height,width | - | D['images'] |

#### More concepts
+ IOU
+ precision and recall:
  + TP: conf>=confThreshold and IOU>=iouThreshold
  + TN: repeat detect or conf<confThreshold and IOU>=iouThreshold
  + FN: IOU<iouThreshold
+ precision-recall curve by confThreshold
+ AP = area under "smooth" PR-curve e.g. 1 if precision=recall=1
+ mAP = average AP among categories
  + mAP50: (default) iouThreshold=0.5
  + mAP75: iouThreshold=0.75
