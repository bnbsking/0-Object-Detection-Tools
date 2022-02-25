import glob, os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, pathList, annotList, idList):
        self.pathList = pathList
        self.annotList = annotList
        self.idList = idList
    def __len__(self):
        return len(self.pathList)
    def __getitem__(self, index):
        imgPath, antPath = self.pathList[index], self.annotList[index]
        x = cv2.imread( imgPath )
        height, width, _ = x.shape # 1080,1920,3
        if abs(height-800)>=abs(width-800): # far side scale to 800 # out 450,800,3
            x = cv2.resize( x,(int(width*800/height,800)) ) # remember to reverse order
        else:
            x = cv2.resize( x,(800,int(height*800/width)) )        
        x = x[:,:,::-1]
        x = (x/255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        x = np.moveaxis(x,-1,0) # 3,450,800
        x = torch.Tensor(x)#.to(self.device)
        #
        boxes, labels, area = [], [], []
        for line in open(antPath,"r").readlines():
            cid, cx, cy, w, h = line.split(" ")
            boxes.append([float(cx),float(cy),float(w),float(h)])
            labels.append(int(cid))
            area.append( float(w)*float(h)*x.shape[0]*x.shape[1] )
        D = {\
             "boxes":torch.Tensor(boxes), #.to(self.device),
             "labels":torch.Tensor(labels).long(), #.to(self.device),
             "image_id":torch.Tensor([self.idList[index]]), #.to(self.device),
             "area":torch.Tensor(area), #.to(self.device),
             "iscrowd":torch.Tensor([0]*len(boxes)), #.to(self.device),
             "orig_size":torch.Tensor([height,width]), #.to(self.device),
             "size":torch.Tensor([x.shape[0],x.shape[1]]), #.to(self.device),
            }
        return x, D

blackList = ["1029_1112_00267"]
def my_datasets(path="/home/jovyan/data-vol-1/detreg/data/ilsvrc100/train/2d", ratio=0.8): # yoloFloat format
    imgPreS, antPreS = set(), set()
    for imgPath in glob.glob(path+"/*.jpg"):
        imgPre  = imgPath.split("/")[-1].split(".")[0]
        if imgPre not in blackList:
            imgPreS.add(imgPre)
    for antPath in glob.glob(path+"/*.txt"):
        antPre  = antPath.split("/")[-1].split(".")[0]
        if antPre not in blackList:
            antPreS.add(antPre)
    intPreS = imgPreS.intersection(antPreS)
    print(f"len(imgPretextSet)={len(imgPreS)}\nlen(antPretextSet)={len(antPreS)}\nlen(intPretextSet)={len(intPreS)}")
    
    imgPathList = list(map(lambda s:f"{path}/{s}.jpg", sorted(list(intPreS)) ))
    antPathList = list(map(lambda s:f"{path}/{s}.txt", sorted(list(intPreS)) ))
    n = int(len(imgPathList)*ratio)
    trainPathList, trainAnnotPathList = imgPathList[:n], antPathList[:n]
    valPathList, valAnnotPathList = imgPathList[n:], antPathList[n:]
    device = torch.device('cuda')
    train_dataset = MyDataset(trainPathList,trainAnnotPathList,range(0,n))
    val_dataset   = MyDataset(valPathList,valAnnotPathList,range(n,len(intPreS)))
    print("train dataset path example:", train_dataset.pathList[0])
    print("valid dataset path example:", val_dataset.pathList[0])
    return train_dataset, val_dataset

my_datasets()
