#!/bin/bash

python test.py --data ./_code/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights ./runs/train/_yolov7/weights/best.pt --name yolov7_640_val