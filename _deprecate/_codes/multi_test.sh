#!/bin/bash

WEIGHT_DIR="runs/train/yolov711/weights"

for WEIGHT in $(ls $WEIGHT_DIR | grep 'epoch_2[0-9][0-9][0-9]\|best.pt\|best_[0-9][0-9][0-9][0-9]'); do
    WEIGHT=./$WEIGHT_DIR/$WEIGHT
    echo ---$WEIGHT---
    #echo ---$WEIGHT--- >> ./_codes/eval.txt
    #python test.py --data ./_codes/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights $WEIGHT --name multi_eval_best >> ./_codes/eval.txt
done

# grep 'epoch_2[0-9][0-9][0-9]\|best'
