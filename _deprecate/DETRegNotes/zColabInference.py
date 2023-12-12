from models import build_model
from argparse import Namespace
from datasets.coco import make_coco_transforms
import torch
from PIL import Image
import requests
import time, os, glob

# load model
args = {'lr': 0.0002, 'max_prop': 30, 'lr_backbone_names': ['backbone.0'], 'lr_backbone': 2e-05, 'lr_linear_proj_names': ['reference_points', 'sampling_offsets'], 'lr_linear_proj_mult': 0.1, 'batch_size': 4, 'weight_decay': 0.0001, 'epochs': 50, 'lr_drop': 40, 'lr_drop_epochs': None, 'clip_max_norm': 0.1, 'sgd': False, 'filter_pct': -1, 'with_box_refine': False, 'two_stage': False, 'strategy': 'topk', 'obj_embedding_head': 'intermediate', 'frozen_weights': None, 'backbone': 'resnet50', 'dilation': False, 'position_embedding': 'sine', 'position_embedding_scale': 6.283185307179586, 'num_feature_levels': 4, 'enc_layers': 6, 'dec_layers': 6, 'dim_feedforward': 1024, 'hidden_dim': 256, 'dropout': 0.1, 'nheads': 8, 'num_queries': 300, 'dec_n_points': 4, 'enc_n_points': 4, 'pretrain': '', 'load_backbone': 'swav', 'masks': False, 'aux_loss': True, 'set_cost_class': 2, 'set_cost_bbox': 5, 'set_cost_giou': 2, 'object_embedding_loss_coeff': 1, 'mask_loss_coef': 1, 'dice_loss_coef': 1, 'cls_loss_coef': 2, 'bbox_loss_coef': 5, 'giou_loss_coef': 2, 'focal_alpha': 0.25, 'dataset_file': 'coco', 'dataset': 'imagenet', 'data_root': 'data', 'coco_panoptic_path': None, 'remove_difficult': False, 'output_dir': '', 'cache_path': 'cache/ilsvrc/ss_box_cache', 'device': 'cuda', 'seed': 42, 'resume': '', 'eval_every': 1, 'start_epoch': 0, 'eval': False, 'viz': False, 'num_workers': 2, 'cache_mode': False, 'object_embedding_loss': False, "model": "deformable_detr"}
args = Namespace(**args)
model, criterion, postprocessors = build_model(args)
model = model.to("cuda")
#checkpoint = torch.hub.load_state_dict_from_url("https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_imagenet.pth", progress=True, ) # map_location=torch.device('cpu')
checkpoint = torch.load("exps/DETReg_top30_in/checkpoint.pth")
load_msg = model.load_state_dict(checkpoint['model'], strict=False)
transforms = make_coco_transforms('val')

# get inference time
img_url = "https://media.ktoo.org/2013/10/Brown-Bears.jpg"
im = Image.open(requests.get(img_url, stream=True).raw)
im_t, _ = transforms(im, None) # img_t: (3,800,1208) Tensor # im_t.unsqueeze(0): (1,3,800,1208) Tensor
res = model( im_t.unsqueeze(0).to("cuda") )
start = time.time()
res = model( im_t.unsqueeze(0).to("cuda") ) # res['pred_logits']:(1,300,91) # res['pred_logits'][...,1]:(1,300) last row, inverse sigmoid confidence
print(time.time()-start)
#scores = torch.sigmoid(res['pred_logits'][..., 1]) # (1,300) score
#pred_boxes = res['pred_boxes']

# testing
from util.box_ops import box_cxcywh_to_xyxy
from util.plot_utils import plot_results
from matplotlib import pyplot as plt
import numpy as np

class boxAny2Voc:
    def voc(xmin,ymin,xmax,ymax,width=None,height=None):
        return int(xmin),int(ymin), int(xmax), int(ymax)
    def yoloFloat(cx,cy,w,h,width=None,height=None): # width, height only valid
        xmin = int((float(cx)-float(w)/2)*float(width))
        ymin = int((float(cy)-float(h)/2)*float(height))
        xmax = int((float(cx)+float(w)/2)*float(width))
        ymax = int((float(cy)+float(h)/2)*float(height))
        return xmin, ymin, xmax, ymax
    def yoloInt(cx,cy,w,h,width=None,height=None):
        xmin = int(int(cx)-int(w)/2)
        ymin = int(int(cy)-int(h)/2)
        xmax = int(int(cx)+int(w)/2)
        ymax = int(int(cy)+int(h)/2)
        return xmin, ymin, xmax, ymax
    def coco(xmin,ymin,w,h,width=None,height=None):
        return int(xmin), int(ymin), int(xmin)+int(w), int(ymin)+int(h)

def IOU(boxA, boxB): # VOC
    (xminA, yminA, xmaxA, ymaxA), (xminB, yminB, xmaxB, ymaxB) = boxA, boxB
    inter = max(0,min(ymaxA,ymaxB)-max(yminA,yminB)) * max(0,min(xmaxA,xmaxB)-max(xminA,xminB))
    areaA = (ymaxA-yminA) * (xmaxA-xminA)
    areaB = (ymaxB-yminB) * (xmaxB-xminB)
    return inter / (areaA+areaB-inter)

def NMS(bboxes, cfs, boxesType="yoloFloat", threshold=0.3):
    bboxes, cfs = np.array(bboxes), np.array(cfs)
    nmsBoxes, nmsCIDs, nmsCFs = [], [], []
    while len(bboxes)>=2:
        nmsBoxes.append( bboxes[0] )
        nmsCFs.append( cfs[0] )
        boxA = getattr(boxAny2Voc,boxesType)(bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], width=1000, height=1000)
        bboxes, cfs = bboxes[1:], cfs[1:]
        keepI = []
        for i,box in enumerate(bboxes):
            boxB = getattr(boxAny2Voc,boxesType)(box[0], box[1], box[2], box[3], width=1000, height=1000)
            iou  = IOU(boxA,boxB)
            if iou<threshold:
                keepI.append(i)
        bboxes = bboxes[keepI]
        cfs = cfs[keepI]
    return nmsBoxes, nmsCFs

def confThreshold(pred_boxes, scores, threshold):
    i = 0
    while scores[0,i]>=threshold:
        i+=1
    return pred_boxes[:,:i], scores[:,:i]

destFolder = "exps/DETReg_top30_in"
for imgPath in sorted(glob.glob("data/ilsvrc100/train/2d/*.jpg"))[:3]:
    im = Image.open(imgPath)
    im_t, _ = transforms(im, None) # img_t: (3,800,1208) Tensor # im_t.unsqueeze(0): (1,3,800,1208) Tensor
    res = model( im_t.unsqueeze(0).to("cuda") )
    scores = torch.sigmoid(res['pred_logits'][..., 1]) # scores: shape=(1,300), dtype=float(0-1)
    pred_boxes = res['pred_boxes']
    img_w, img_h = im.size
    pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.Tensor([img_w, img_h, img_w, img_h]).to("cuda") # pred_boxes: shape=(1,300,4), dtype=float (VOC)
    
    I = scores.argsort(descending = True) # sorted by model confidence # I: shape=(1,300), dtype=int(0-299)
    pred_boxes = pred_boxes_[0, I[0,:300]] # pick top 3 proposals
    scores = scores[0, I[0,:300]]
    if True:
        print(pred_boxes.shape, scores.shape)
        pred_boxes, scores = NMS(bboxes, scores, boxesType="voc", threshold=0.3)
        print(pred_boxes.shape, scores.shape)
        pred_boxes, scores = confThreshold(bboxes, scores, threshold=0.3)
        print(pred_boxes.shape, scores.shape)
    #print(pred_boxes.detach().cpu().numpy().astype(int))
    #print(scores)
    plt.figure()
    plot_results(np.array(im), scores, pred_boxes, plt.gca(), norm=False)
    plt.axis('off')
    plt.savefig(f"{destFolder}/{imgPath.split('/')[-1]}")
