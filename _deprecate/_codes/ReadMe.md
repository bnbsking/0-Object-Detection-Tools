### Installation
###### from exists environment
1. conda create -n yolo python=3.7
2. cp -r /home/jovyan/data-vol-1/envs/yolo /home/jovyan/.conda/yolo 
3. source /opt/conda/bin/activate yolo
###### from scratch
1. pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
2. pip install tensorboard tqdm
3. pip install numpy matplotlib pandas opencv-python-headless seaborn scipy

### File structures and formats
1. source data:
    + imgs: /home/jovyan/nas-dataset/recycling/backboneData/2022-06-14/\*.jpg
    + labels: /home/jovyan/nas-dataset/recycling/backboneData/2022-06-14_label_tetra_v5/\*.txt
2. data preparation (for both labeled and unlabeled): /home/jovyan/data-vol-1/yolov7/\_data/pcV6/
    + data/\*.txt in yolo format (see Processes 1,7)
    + data/\*.jpg (for labeled only, see Processes 2)
    + train.txt, val.txt (see Processes 2,7)
        + /path/to/above/txtAndJpg\n
3. training weights: /home/jovyan/data-vol-1/yolov7/runs/train/yolov7_pcv6v0
    + results.txt (see Processes 3)
    + weights/\*.pt (see Processes 3)
4. inference result: /home/jovyan/data-vol-1/yolov7/runs/test/pcv6v0/
    + best_851_predictions.json (see Processes 5,8)
        + list[dict] e.g. {"image_id": "20220614_100020_666", "category_id": 1, "bbox": \[589.0, 400.0, 287.0, 217.0\], "score": 0.97461}
    + confusion.jpg, pr.npy, pr.jpg, GT\_\*\_PD\_\*/\*.jpg (see Processes 6)
    + yoloFormat/\*.txt (see Processes 9)
        + cid,cx,cy,w,h,conf

### Processes in \_codes/
1. copyData.ipynb
    + copy \*.jpg and \*.txt from NAS to \_data/pcV6/data
2. preprocessingMulti.ipynb
    + generate \_data/pcV6/train.txt and \_data/pcV6/val.txt
3. train.sh 
    + specify transfer model, epoch, output_dir_name and configs 
    + ./coco.yaml: specify path to train.txt and val.txt
    + ./hyp.scratch.p5.yaml
    + ./yolov7.yaml
    + output result.txt and weights/\*.pt
4. plotLoss.ipynb
    + plot training curve and obtain best checkpoint
5. multi_test.sh
    + evaluate over all checkpoints -> eval.txt. Then get the best weight that performs the best accuracy  
6. result.ipynb
    + obtain p/r/pr curves, confusion matrix and its grid images
    + need to call confusion_matrix.py and visualization.py

<!--inference on unlabeled data-->
7. preprocessingUnlabeled.ipynb
    + generate train.txt and val.txt just like preprocessingMulti.ipynb but the labels are empty txt
8. test.sh
    + evaluate over all unlabeled data
9. json2YoloFormat.ipynb
    + convert \_exps/runs/test/pcv6v0/best_851_predictions.json into \_exps/runs/test/pcv6v0/yoloFormat/\*.txt

<!--Internal cycle or External cycle-->
10. internal cycle, active learning
    + active.ipynb
        + act on prediction of unlabled data and their horizontal flipped, auto copy data
11. external cycle, ensembling
    + compareTwoModels.ipynb
        + compare two models result and output json report
    + compareAutoLabeling_Find_Inconsistent_Inferences.ipynb
        + compare three models result and output csv report
    + compareByCsv.ipynb
        + read csv report above and auto copy problem data
12. more dubugging:
    + visualizationDebugging.ipynb
        + visualize previous and next images
    + label_IOU_checking.ipynb
        + check high iou labels
