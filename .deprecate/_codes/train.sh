#!/bin/bash

python train.py --workers 8 --device 0 --batch-size 16 --data ./_code/coco.yaml --img 640 640 --cfg ./_code/yolov7.yaml --weights '' --name yolov7 --hyp ./_code/hyp.scratch.p5.yaml #--resume

#python train.py --workers 8 --device 0 --batch-size 16 --data ./_code/coco.yaml --img 640 640 --cfg ./_code/yolov7.yaml --weights ./runs/train/yolov72/weights/epoch_024.pt --name yolov7 --hyp ./_code/hyp.scratch.p5.yaml
