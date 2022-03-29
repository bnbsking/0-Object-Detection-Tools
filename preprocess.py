import os, glob
import cv2

def getInterPrefix(glob1, glob2, blackS={}, show=True):
    path2prefix = lambda path:path.split('/')[-1].split('.')[0]
    notInBlackS = lambda prefix:prefix not in blackS 
    prefix1S = set( filter(notInBlackS, map(path2prefix,glob.glob(glob1))) )
    prefix2S = set( filter(notInBlackS, map(path2prefix,glob.glob(glob2))) )
    prefixL  = sorted(list(prefix1S.intersection(prefix2S)))
    if show:
        print( f"len(prefix1S)={len(prefix1S)}, len(prefix2S)={len(prefix2S)}, len(prefix)={len(prefixL)}" )
    return prefixL

def yoloFloatCheck(imgFolder, antFolder, prefixL):
    print("get shapeD (shape->prefixL), classD (cid->prefixL), outBoundS (prefix)")
    shapeD, classD, outBoundS = {}, {}, set()
    for i,prefix in enumerate(prefixL):
        print(f"\r{i+1}/{len(prefixL)}", end="")
        img = cv2.imread(f"{imgFolder}/{prefix}.jpg")
        shape = img.shape if hasattr(img,"shape") else None
        shapeD[shape] = shapeD[shape]+[prefix] if shape in shapeD else [prefix]
        with open(f"{antFolder}/{prefix}.txt","r") as f:
            for line in f.readlines():
                cid, cx, cy, w, h = line.split(" ")
                classD[cid] = classD[cid]+[prefix] if cid in classD else [prefix]
                if not 0<=float(cx)<=1 or not 0<=float(cy)<=1 or not 0<=float(w)<=1 or not 0<=float(h[:-1])<=1:
                    outBoundS.add(prefix)
    shapeDItemLen = { shape:len(shapeD[shape]) for shape in shapeD}
    classDItemLen = { cid:len(classD[cid]) for cid in classD }    
    print("\n", f"shapeDItemLen={shapeDItemLen}, classDItemLen={classDItemLen}, len(outBoundS)={len(outBoundS)}")
    return shapeD, classD, outBoundS

def cleanData(imgFolder, antFolder, imgFolderDest, antFolderDest, prefixL):
    os.makedirs(imgFolderDest, exist_ok=True)
    os.makedirs(antFolderDest, exist_ok=True)
    for i,prefix in enumerate(prefixL):
        print(f"\r{i+1}/{len(prefixL)}", end="")
        os.system(f"cp {imgFolder}/{prefix}.jpg {imgFolderDest} && cp {antFolder}/{prefix}.txt {antFolderDest}")

def mAP_format(imgFolder, antFolder, imgFolderDest, antFolderDest, prefixL):
    os.makedirs(imgFolderDest, exist_ok=True)
    os.makedirs(antFolderDest, exist_ok=True)
    for i,prefix in enumerate(prefixL):
        print(f"\r{i+1}/{len(prefixL)}", end="")
        os.system(f"cp {imgFolder}/{prefix}.jpg {imgFolderDest}")
        height, width, _ = cv2.imread(f"{imgFolder}/{prefix}.jpg").shape
        with open(f"{antFolder}/{prefix}.txt",'r') as f:
            lines = f.readlines()
        with open(f"{antFolderDest}/{prefix}.txt",'w') as f:
            for line in lines:
                cid, cx, cy, w, h = line.replace('\n','').split(" ")
                xmin = int((float(cx)-float(w)/2)*width)
                ymin = int((float(cy)-float(h)/2)*height)
                xmax = int((float(cx)+float(w)/2)*width)
                ymax = int((float(cy)+float(h)/2)*height)
                f.write(f"{cid} {xmin} {ymin} {xmax} {ymax}\n")
    print("\n", len(os.listdir(imgFolderDest)), len(os.listdir(antFolderDest)) )
    
"""
import sys
ppath = "/home/jovyan/data-vol-1/detreg/zPackages"
sys.path += [] if ppath in sys.path else [ppath] 
import preprocess as pre

prefixL = pre.getInterPrefix("/home/jovyan/data-vol-2/recycling/Lab/test_v2/*.jpg","/home/jovyan/data-vol-2/recycling/Lab/test_v2/*.txt")

shapeD,classD,outBoundS = pre.yoloFloatCheck("/home/jovyan/data-vol-2/recycling/Lab/test_v2/", "/home/jovyan/data-vol-2/recycling/Lab/test_v2/", prefixL)

pre.cleanData("/home/jovyan/data-vol-2/recycling/Lab/test_v2/", "/home/jovyan/data-vol-2/recycling/Lab/test_v2/", "xtest", "xtest", prefixL)

shapeD,classD,outBoundS = pre.yoloFloatCheck("xtest", "xtest", prefixL)

pre.mAP_format("xtest","xtest","yoloIntImg","yoloIntAnt",prefixL)
"""
