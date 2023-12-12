git clone https://github.com/amirbar/detreg

Download Colab pretask inference code:

https://colab.research.google.com/drive/1ByFXJClyzNVelS7YdT53_bMbwYeMoeNb?usp=sharing

Environment:
+ A100, Ubuntu 20.04, Python 3.7.11
+ torch 1.8.1+cu111
+ Compiling CUDA operators
+ pip install -r requirements.txt

Download dataset to ./data in standard format: ImageNet100, MSCoco

Create code:
+ pretask:
    + load_simsiam.py (trained by lightly)    https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html (customize model I/O and trainingPlot)
    + plotLoss.ipynb
+ downstream:
    + plotLoss.ipynb
    + visualization.ipynb
    + convert.ipynb
    + myDataset.py

Modify scripts:
+ models/_init_.py - build_model: num_classes
+ engine.py - viz:
    + break condition
    + print overall confidence, classIDs, bboxes.
    + Shape revision for raw Coco dataset
+ engine.py - train_one_epoch:
    + evaluate coco (downstream only)
+ main.py:
    + replace swav_model
    + replace dataset (downstream only)
    ```python
    #dataset_train, dataset_val = get_datasets(args)
    if False:
        model.load_state_dict(torch.load("exps/official_weights/checkpoint_coco.pth"),strict=False)
    if True:
        from zDataset import my_datasets
        dataset_train, dataset_val, dataset_test = my_datasets(trainValPath="data/labv2/train/clean2dr", testPath="data/labv2/test/clean2dr", test=False)
        args.viz = False```

Training and visualization:
+ Pretrain:
    python -u main.py --output_dir exps/DETReg_top30_in --dataset imagenet100 --strategy topk --load_backbone swav --max_prop 30 --object_embedding_loss --lr_backbone 0 --epochs 60 --batch_size 24 --num_workers 1
+ Pretrain Inference:
    python zColabInference.py
+ Finetune (args.viz=False):
    python -u main.py --output_dir exps/DETReg_fine_tune_full_coco --dataset coco --pretrain exps/DETReg_top30_in/checkpoint.pth --batch_size 4 --num_workers 1 --epochs 60
+ Finetune Inference (args.viz=True):
    python -u main.py --output_dir exps/DETReg_fine_tune_full_coco --dataset coco --pretrain exps/DETReg_fine_tune_full_coco/checkpoint.pth --batch_size 1 --num_workers 1

AP Calculation: https://github.com/Cartucho/mAP
+ GT stored in a folder with *.txt
+ Result stored in a folder with *.txt
+ format: “class conf xmin ymin xmax ymax\n“
+ python main.py -na
+ ranking.ipynb
