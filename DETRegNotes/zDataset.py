import glob, os, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, pathList, annotList, idList):
        self.pathList = pathList
        self.annotList = annotList
        self.idList = idList
        #self.more()
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
    
    def more(self):
        self.getImgIds = lambda : self.idList
        self.CatIds = lambda : [0,1,2,3,4]
        self.dataset = {"images":[], "categories":[]}
        for path,id in zip(self.pathList,self.idList):
            x = cv2.imread( path )
            imgD = {'license':0, 'file_name':path.split('/')[-1], 'coco_url':'', 'height':x.shape[0], 'width':x.shape[1], 'date_captured':'', 'flickr_url':'', 'id':id}
            self.dataset['images'].append( imgD )
        for id,name in zip(range(5),['PLC','nPLC','PAC','nPAC','FOIL']):
            self.dataset['categories'].append( {'supercategory':'trash', 'id':id, 'name':name} )
        from pycocotools.coco import COCO
        self.coco = COCO("data/labv2coco/labels.json")
        self.getAnnIds = self.coco.getAnnIds
        self.loadAnns  = self.coco.loadAnns

blackList = ["1029_1112_00267"]
def my_datasets(trainValPath="data/labv2/train/2d", testPath="data/labv2/test/2d", ratio=0.8): # includes .jpg and .txt in yolo format
    
    def getIntersectionPretext(path):
        imgPretextS, antPretextS = set(), set()
        for imgPath in glob.glob(path+"/*.jpg"):
            imgPretext = imgPath.split("/")[-1].split(".")[0]
            if imgPretext not in blackList:
                imgPretextS.add(imgPretext)
        for antPath in glob.glob(path+"/*.txt"):
            antPretext = antPath.split("/")[-1].split(".")[0]
            if antPretext not in blackList:
                antPretextS.add(antPretext)
        pretextS = imgPretextS.intersection(antPretextS)
        pretextL = sorted(list(pretextS))
        print(len(pretextL))
        return pretextL
    
    trainValL, testL = getIntersectionPretext(trainValPath), getIntersectionPretext(testPath) if testPath else []
    random.Random(4).shuffle(trainValL)
    
    trainValL= trainValL[:int(1*len(trainValL))] # 1, 0.2, 0.05
    trainValImgPathL = list(map(lambda s:f"{trainValPath}/{s}.jpg", list(trainValL) ))
    trainValAntPathL = list(map(lambda s:f"{trainValPath}/{s}.txt", list(trainValL) ))
    testImgPathL = list(map(lambda s:f"{testPath}/{s}.jpg", list(testL) ))
    testAntPathL = list(map(lambda s:f"{testPath}/{s}.txt", list(testL) ))
    
    n = int(len(trainValL)*ratio)
    trainImgPathL, valImgPathL = trainValImgPathL[:n], trainValImgPathL[n:]
    trainAntPathL, valAntPathL = trainValAntPathL[:n], trainValAntPathL[n:]
    
    train_dataset = MyDataset(trainImgPathL,trainAntPathL,range(0,n))
    val_dataset   = MyDataset(valImgPathL,valAntPathL,range(n,len(trainValL)))
    test_dataset  = MyDataset(testImgPathL,testAntPathL,range(len(testL))) if testPath else val_dataset
    if "test":
        val_dataset = test_dataset
    
    print("train dataset path example:", train_dataset.pathList[:3])
    print("valid dataset path example:", val_dataset.pathList[:3])
    print("test dataset path example:", test_dataset.pathList[:3])
    print(f"len(train_dataset)={len(train_dataset)}")
    print(f"len(val_dataset)={len(val_dataset)}")
    print(f"len(test_dataset)={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

#my_datasets()
