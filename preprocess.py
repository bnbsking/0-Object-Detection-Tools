import glob, os
import cv2

class Preprocess:
    def __init__(self, imgFolder, antFolder=None, blackS=set()): # img type is .jpg; txt type is .txt
        self.imgFolder = imgFolder
        self.antFolder = antFolder if antFolder else imgFolder
        self.blackS    = blackS
        self.getInterPrefix()
    
    def getInterPrefix(self):
        imgPrefix  = map(lambda path:path.split('/')[-1].split('.')[0], glob.glob(f"{self.imgFolder}/*.jpg"))
        antPrefix  = map(lambda path:path.split('/')[-1].split('.')[0], glob.glob(f"{self.antFolder}/*.txt"))
        imgPrefixS = set(filter(lambda prefix:prefix not in self.blackS,imgPrefix))
        antPrefixS = set(filter(lambda prefix:prefix not in self.blackS,antPrefix))
        self.prefixL = sorted(list(imgPrefixS.intersection(antPrefixS)))
        print( f"len(imgPrefixS)={len(imgPrefixS)}, len(antPrefixS)={len(antPrefixS)}, len(prefixL)={len(self.prefixL)}" )
    
    def check(self): # yoloFloat only
        print("collect prefixes by 'same shapes', 'category' and 'out of boundary'")
        shapeD, classD, outBoundS = {}, {}, set() # shape->prefixL, cid->prefixL, prefix
        for i,prefix in enumerate(self.prefixL):
            print(f"\r{i+1}/{len(self.prefixL)}", end="")
            img = cv2.imread(f"{self.imgFolder}/{prefix}.jpg")
            shape = img.shape if hasattr(img,"shape") else None
            shapeD[shape] = shapeD[shape]+[prefix] if shape in shapeD else [prefix]
            with open(f"{self.antFolder}/{prefix}.txt","r") as f:
                for line in f.readlines():
                    cid, cx, cy, w, h = line.split(" ")
                    classD[cid] = classD[cid]+[prefix] if cid in classD else [prefix]
                    if not 0<=float(cx)<=1 or not 0<=float(cy)<=1 or not 0<=float(w)<=1 or not 0<=float(h[:-1])<=1:
                        outBoundS.add(prefix)
        shapeDItemLen = { shape:len(shapeD[shape]) for shape in shapeD}
        classDItemLen = { cid:len(classD[cid]) for cid in classD } 
        print("\n", f"shapeDItemLen={shapeDItemLen}, classDItemLen={classDItemLen}, len(outBoundS)={len(outBoundS)}")
        return shapeD, classD, outBoundS
    
    def copyData(self, imgFolderDest, antFolderDest=None):
        antFolderDest = antFolderDest if antFolderDest else imgFolderDest
        os.makedirs(imgFolderDest, exist_ok=True)
        os.makedirs(antFolderDest, exist_ok=True)
        for i,prefix in enumerate(self.prefixL):
            print(f"\r{i+1}/{len(self.prefixL)}", end="")
            os.system(f"cp {self.imgFolder}/{prefix}.jpg {imgFolderDest} && cp {self.antFolder}/{prefix}.txt {antFolderDest}")
        print("\ndestination check: ", end=" ")
        Preprocess(imgFolderDest, antFolderDest)

    def make_mAP_format(self, imgFolderDest, antFolderDest):
        os.makedirs(imgFolderDest, exist_ok=True)
        os.makedirs(antFolderDest, exist_ok=True)
        for i,prefix in enumerate(self.prefixL):
            print(f"\r{i+1}/{len(self.prefixL)}", end="")
            os.system(f"cp {self.imgFolder}/{prefix}.jpg {imgFolderDest}")
            height, width, _ = cv2.imread(f"{self.imgFolder}/{prefix}.jpg").shape
            with open(f"{self.antFolder}/{prefix}.txt",'r') as f:
                lines = f.readlines()
            with open(f"{antFolderDest}/{prefix}.txt",'w') as f:
                for line in lines:
                    cid, cx, cy, w, h = line.replace('\n','').split(" ")
                    xmin = int((float(cx)-float(w)/2)*width)
                    ymin = int((float(cy)-float(h)/2)*height)
                    xmax = int((float(cx)+float(w)/2)*width)
                    ymax = int((float(cy)+float(h)/2)*height)
                    f.write(f"{cid} {xmin} {ymin} {xmax} {ymax}\n")
        print("\ndestination check: ", end=" ")
        Preprocess(imgFolderDest, antFolderDest)
