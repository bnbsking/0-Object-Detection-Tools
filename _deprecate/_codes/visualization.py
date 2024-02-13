import re, glob, json, random
import cv2 # imageIO and processing
import matplotlib.pyplot as plt # show
import numpy as np

def stroke(A,L,color): # list[bool] # up, mid, down, upleft, upright, downleft, downright
    if not L:
        A[:3,:3,:] = 0,0,0
        A[-3:,-3:,:] = 0,0,0
        for i in range(10):
            A[i*2:i*2+2,10-i-1:10-i,:] = 0,0,0
    else:
        if L[0]:
            A[0:3,:,:] = 0,0,0
        if L[1]:
            A[10-1:10+1,:,:] = 0,0,0        
        if L[2]:
            A[20-3:20,:,:] = 0,0,0
        if L[3]:
            A[:10,:3,:] = 0,0,0
        if L[4]:
            A[:10,10-3:,:] = 0,0,0
        if L[5]:
            A[10:,:3,:] = 0,0,0
        if L[6]:
            A[10:,10-3:,:] = 0,0,0
    B = np.array([ [color for j in range(5+10+5)] for i in range(5+20+5) ]).astype(float)
    B[5:25,5:15,:] = A
    return B # 30,20

def getImg(n,color=(1,1,1)):
    A = np.array([ [color for j in range(10)] for i in range(20) ]).astype(float)
    D = {0:[1,0,1,1,1,1,1],
         1:[0,0,0,1,0,1,0],
         2:[1,1,1,0,1,1,0],
         3:[1,1,1,0,1,0,1],
         4:[0,1,0,1,1,0,1],
         5:[1,1,1,1,0,0,1],
         6:[1,1,1,1,0,1,1],
         7:[1,0,0,0,1,0,1],
         8:[1,1,1,1,1,1,1],
         9:[1,1,1,1,1,0,1],
         10:[], # percent
        }
    return stroke(A,D[n],color)

def getPatch(a,b,color=(1,1,1)):
    A = np.array([ [color for j in range(20+20+20)] for i in range(30) ]).astype(float)
    A[:,:20,:] = getImg(a,color)
    A[:,20:40,:] = getImg(b,color)
    A[:,40:60,:] = getImg(10,color)
    return A # 30,60

if True:
    color = [ (1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (0,0,0), (0.5,0.5,0.5) ]
else:
    color = set()
    while len(color)<90:
        color.add( tuple([round(random.random(),2) for i in range(3)]) )
    color = list(color)

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

def getAnnot(imgPath, annotPath, classList):
    if ".xml" in annotPath: # Pascal VOC
        xml = open(annotPath,"r").read()
        nameL = re.findall("<name>(.*)</name>",xml)
        xminL = re.findall("<xmin>(.*)</xmin>",xml)
        yminL = re.findall("<ymin>(.*)</ymin>",xml)
        xmaxL = re.findall("<xmax>(.*)</xmax>",xml)
        ymaxL = re.findall("<ymax>(.*)</ymax>",xml)
        boxes, cids = [], []        
        for name,xmin,ymin,xmax,ymax in zip(nameL,xminL,yminL,xmaxL,ymaxL):
            cids.append( classList.index(name) )
            xmin, ymin, xmax, ymax = boxAny2Voc.voc(xmin,ymin,xmax,ymax)
            boxes.append( [xmin, ymin, xmax, ymax] )
        return boxes, cids
    elif ".txt" in annotPath: # YOLO
        height, width, _ = cv2.imread(imgPath).shape
        boxes, cids = [], []
        for line in open(annotPath,"r").readlines():
            cid, cx, cy, w, h = line.split(" ")
            cids.append( int(cid) )
            if "." in cx:
                xmin, ymin, xmax, ymax = boxAny2Voc.yoloFloat(cx, cy, w, h, width, height)
            else:
                xmin, ymin, xmax, ymax = boxAny2Voc.yoloInt(cx, cy, w, h)
            boxes.append( [xmin, ymin, xmax, ymax] )
        return boxes, cids
    elif ".json" in annotPath: # COCO
        D = json.load( open(annotPath,"r") )
        imgName   = imgPath.split('/')[-1]
        imgDict   = list(filter(lambda d:d["file_name"]==imgName,D["images"]))[0]
        id        = imgDict["id"]
        annotDict = list(filter(lambda d:d["image_id"]==id,D["annotations"]))
        boxes, cids = [], []
        for d in annotDict:
            cid = d['category_id']#-1
            cids.append( cid )
            xmin, ymin, w, h = d['bbox']
            xmin, ymin, xmax, ymax = boxAny2Voc.coco(xmin, ymin, w, h) 
            boxes.append( [xmin, ymin ,xmax, ymax] )
        return boxes, cids
    else:
        raise ValueError("Annotation not found")

def show(imgPath, annotPath, boxesTypePD=None, boxesPD=None, cidsPD=None, cfsPD=None, classList=None, savePath=None, valueRatios=(1,1)):
    imgRaw = cv2.imread(imgPath)[:,:,::-1]/255

    if not annotPath:
        imgGT = np.zeros((imgRaw.shape[0],imgRaw.shape[1],3))
    else:
        imgGT = imgRaw.copy()
        boxesGT, cidsGT = getAnnot(imgPath,annotPath,classList) # boxesType, boxes, cids
        for (xmin,ymin,xmax,ymax),cid in zip(boxesGT,cidsGT):
            xmin, ymin, xmax, ymax = max(xmin,4), max(ymin,4), min(xmax,imgRaw.shape[1]-4), min(ymax,imgRaw.shape[0]-4)
            imgGT[ymin-4:ymin+4,xmin:xmax,:] = color[cid]
            imgGT[ymax-4:ymax+4,xmin:xmax,:] = color[cid]
            imgGT[ymin:ymax,xmin-4:xmin+4,:] = color[cid]
            imgGT[ymin:ymax,xmax-4:xmax+4,:] = color[cid]
        
    if not boxesTypePD:
        imgPD = np.zeros((imgRaw.shape[0],imgRaw.shape[1],3))
    else:
        imgPD = imgRaw.copy()
        height, width, _ = imgPD.shape
        for i,(b1,b2,b3,b4) in reversed(list(enumerate(boxesPD))):
            xmin, ymin, xmax, ymax = getattr(boxAny2Voc,boxesTypePD)(b1,b2,b3,b4,width,height)
            xmin, ymin, xmax, ymax = max(xmin,4), max(ymin,4), min(xmax,imgRaw.shape[1]-4), min(ymax,imgRaw.shape[0]-4)
            imgPD[ymin-4:ymin+4,xmin:xmax,:] = color[cidsPD[i]]
            imgPD[ymax-4:ymax+4,xmin:xmax,:] = color[cidsPD[i]]
            imgPD[ymin:ymax,xmin-4:xmin+4,:] = color[cidsPD[i]]
            imgPD[ymin:ymax,xmax-4:xmax+4,:] = color[cidsPD[i]]
            # percentage patches
            ud, td = int(cfsPD[i]*10), int(cfsPD[i]*100)%10
            P = getPatch(ud,td,color=color[cidsPD[i]])
            (ph, pw, _), (rh,rw) = P.shape, valueRatios
            P = cv2.resize( P, (int(pw*rw),int(ph*rh)) )
            try:
                if ymin>=P.shape[0] and xmin+P.shape[1]<imgPD.shape[1]: # upper bar
                    imgPD[ymin-P.shape[0]:ymin,xmin:xmin+P.shape[1],:] = P
                elif ymax+P.shape[0]<imgPD.shape[0] and xmin+P.shape[1]<imgPD.shape[1]: # down bar
                    imgPD[ymax:ymax+P.shape[0],xmin:xmin+P.shape[1],:] = P
            except:
                pass

    fig = plt.figure(figsize=(20,10))
    fig.set_facecolor("white")

    plt.subplot(1,2,1)
    plt.title(f"GT - {imgPath.split('/')[-1]}", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    for r,g,b in color:
        c2hex = lambda c: ("0" if c<=1/16 else '') + hex(int(c*255))[2:]
        plt.scatter([0],[0],c=f"#{c2hex(r)}{c2hex(g)}{c2hex(b)}")
    plt.legend(labels=classList, fontsize=16, framealpha=0.25)
    plt.imshow(imgGT)
    
    plt.subplot(1,2,2)
    plt.title(f"Pred - {imgPath.split('/')[-1]}", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    for r,g,b in color:
        c2hex = lambda c: ("0" if c<=1/16 else '') + hex(int(c*255))[2:]
        plt.scatter([0],[0],c=f"#{c2hex(r)}{c2hex(g)}{c2hex(b)}")
    plt.imshow(imgPD)
    if savePath:
        plt.savefig( f"{savePath}/{imgPath.split('/')[-1]}" )
    else:
        plt.show()
    plt.close()
