git clone https://github.com/WongKinYiu/PyTorch_YOLOv4 # Do not use PT10
conda create -n pt9 python=3.6
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

mkdir zMyData && put data folder into this folder
dataVisualization.ipynb
convert.ipynb

vim coco.yaml
vim coco.name
vim cfg/yolov4-pacsp.cfg -> classes=class and filters=(class+5)*3

vim train.py line 424-427 adjust saving frequency
python3 train.py --device 0 --batch-size 1 --img 640 640 --data data/coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights '' --name yolov4-pacsp --epochs 1
(copy weight from runs/train/yolov4-pacsp/weights)

vim test.py -> line 106: if True: x, y, w, h, conf = float(x), float(y), float(w), float(h), float(conf)
python3 test.py --device 0 --batch 1 --img 640 --conf 0.001 --data data/coco.yaml --cfg cfg/yolov4-pacsp.cfg --weights weights/best.pt
(Get bounding box in test.py line 224 “plot_images(img, targets, paths, f, names)”. Targets.shape = bz*(class,conf,cx,cy,w,h))
