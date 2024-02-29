import os, glob, re, json
import cv2

"""
coco hard to search
yolo lack of raw size -> transfer from yolo to others need height and width
voc lack of classL -> transfer from voc to others need classL
"""

if True:
    xml0="\
<annotation>\n\
	<folder>folderX</folder>\n\
	<filename>filenameX</filename>\n\
	<path>pathX</path>\n\
	<source>\n\
		<database>Unknown</database>\n\
	</source>\n\
	<size>\n\
		<width>widthX</width>\n\
		<height>heightX</height>\n\
		<depth>3</depth>\n\
	</size>\n\
	<segmented>0</segmented>\n\
" # folderX, filenameX, pathX, widthX, heightX need to be replaced
    obj0="\
	<object>\n\
		<name>nameX</name>\n\
		<pose>Unspecified</pose>\n\
		<truncated>0</truncated>\n\
		<difficult>0</difficult>\n\
		<bndbox>\n\
			<xmin>xminX</xmin>\n\
			<ymin>yminX</ymin>\n\
			<xmax>xmaxX</xmax>\n\
			<ymax>ymaxX</ymax>\n\
		</bndbox>\n\
	</object>\n\
" # nameX, xmin, ymin, xmax, ymax need to be replaced
    end0="</annotation>"

def bnce(path, ext=None): # get basename and convert extension
    bs, raw_ext = os.path.basename(path).split(".")
    return bs + "." + (ext if ext else raw_ext)
    
def boxAny2Voc(src_type, b1, b2, b3, b4, width=None, height=None): # width & height yolo only
    if src_type=="voc": # b1,b2,b3,b4 = xmin,ymin,xmax,ymax
        xmin, ymin, xmax, ymax = int(b1), int(b2), int(b3), int(b4)
    elif src_type=="yolo": # b1,b2,b3,b4 = cx,cy,w,h
        xmin = int((float(b1)-float(b3)/2)*float(width))
        ymin = int((float(b2)-float(b4)/2)*float(height))
        xmax = int((float(b1)+float(b3)/2)*float(width))
        ymax = int((float(b2)+float(b4)/2)*float(height))
    elif src_type=="coco": # b1,b2,b3,b4 = xmin,ymin,w,h
        xmin, ymin, xmax, ymax = int(b1), int(b2), int(b1)+int(b3), int(b2)+int(b4)
    else:
        raise KeyError(f"{src_type} Not found")
    return xmin, ymin, xmax, ymax

def boxVoc2Any(des_type, xmin, ymin, xmax, ymax, width=None, height=None): # width & height yolo only
    if des_type=="voc":
        return int(xmin), int(ymin), int(xmax), int(ymax)
    elif des_type=="yolo":
        cx = round((int(xmin)+int(xmax))/2/float(width),6)
        cy = round((int(ymin)+int(ymax))/2/float(height),6)
        w  = round((int(xmax)-int(xmin))/float(width),6)
        h  = round((int(ymax)-int(ymin))/float(height),6)
        return cx, cy, w, h
    elif des_type=="coco":
        xmin = int(xmin)
        ymin = int(ymin)
        w    = int(xmax)-int(xmin)
        h    = int(ymax)-int(ymin)
        return xmin, ymin, w, h
    else:
        raise KeyError(f"{des_type} Not found")

### following are conversions ###

def voc2yolo(sourceFolder, destFolder, classL):
    """
    Convert labels from voc to yolo.

    sourceFolder: str. Path of source that contains *.jpg and *.xml.
    destFolder: str. Path of destination.
    classL: list[str]. list of class names.
    """

    os.makedirs(destFolder, exist_ok=True)
    with open(f"{destFolder}/classes.txt","w") as f:
        for c in classL:
            f.write(f"{c}\n")
    sourceL = glob.glob(f"{sourceFolder}/*.xml")
    for i,xmlPath in enumerate(sourceL):
        print(f"\r{i+1}/{len(sourceL)}", end='')
        xml = open(xmlPath,"r").read()
        width  = int(re.findall("<width>([0-9]*)</width>",xml)[0])
        height = int(re.findall("<height>([0-9]*)</height>",xml)[0])
        nameL = re.findall("<name>(.*)</name>",xml)
        xminL = re.findall("<xmin>(.*)</xmin>",xml)
        yminL = re.findall("<ymin>(.*)</ymin>",xml)
        xmaxL = re.findall("<xmax>(.*)</xmax>",xml)
        ymaxL = re.findall("<ymax>(.*)</ymax>",xml)
        with open(f"{destFolder}/{bnce(xmlPath,'txt')}", "w") as f:
            for name,xmin,ymin,xmax,ymax in zip(nameL,xminL,yminL,xmaxL,ymaxL):
                cid = classL.index(name)
                cx, cy, w, h = boxVoc2Any("yolo",xmin,ymin,xmax,ymax,width,height) #
                pad = lambda s:str(s)+'0'*(8-len(str(s)))
                f.write(f"{cid} {pad(cx)} {pad(cy)} {pad(w)} {pad(h)}\n")

def voc2coco(sourceFolder, destPath, classL):
    """
    Convert labels from voc to coco.

    sourceFolder: str. Path of source that contains *.jpg and *.xml.
    destFolder: str. Path of destination.
    classL: list[str]. list of class names.
    """

    D = {"images":[], "annotations":[], "categories": []}
    D["categories"] = [ {"supercategory":"none","id":i,"name":className} for i,className in enumerate(classL,0) ] # index start from 0
    sourceL = sorted(glob.glob(f"{sourceFolder}/*.xml"))
    boxId = 0 # annotation.id
    for id,xmlPath in enumerate(sourceL):
        print(f"\r{id+1}/{len(sourceL)}", end='')
        xml = open(xmlPath,"r").read()
        filename = bnce(xmlPath,'jpg')
        height = int(re.findall("<height>([0-9]*)</height>",xml)[0])
        width  = int(re.findall("<width>([0-9]*)</width>",xml)[0])
        nameL  = re.findall("<name>(.*)</name>",xml)
        xminL  = re.findall("<xmin>(.*)</xmin>",xml)
        yminL  = re.findall("<ymin>(.*)</ymin>",xml)
        xmaxL  = re.findall("<xmax>(.*)</xmax>",xml)
        ymaxL  = re.findall("<ymax>(.*)</ymax>",xml)
        D["images"].append( {"file_name":filename,"height":height,"width":width,"id":id} )
        for name,xmin,ymin,xmax,ymax in zip(nameL,xminL,yminL,xmaxL,ymaxL):
            xmin, ymin, w, h = boxVoc2Any("coco",xmin,ymin,xmax,ymax) #
            cid= classL.index(name)+0 # index start from 0 
            D["annotations"].append( {"area":w*h,"iscrowd":0,"bbox":[xmin,ymin,w,h],"category_id":cid,"ignore":0,"segmentation":[],"image_id":id,"id":boxId} )
            boxId+=1
    os.makedirs(os.path.dirname(destPath), exist_ok=True)
    with open(destPath, "w") as f:
        json.dump(D,f)

def yolo2voc(sourceFolder, destFolder, classL=None, defaultAspect=None):
    """
    Convert labels from yolo to voc.

    sourceFolder: str. Path of source that contains *.jpg and *.txt.
    destFolder: str. Path of destination.
    classL: list[str] or None. list of class names. If 'classes.txt' is in sourceFolder, this arg can be None.
    defaultAspect: (int,int). Can be specified if all the images have same shape, so the box can be compute without reading all images. 
    """

    global xml0, obj0, end0
    os.makedirs(destFolder, exist_ok=True)
    classL = classL if classL else [ line.replace('\n','') for line in open(f"{sourceFolder}/classes.txt","r").readlines() ]
    sourceL = glob.glob(f"{sourceFolder}/*.txt")
    for i,txtPath in enumerate(sourceL):
        print(f"\r{i+1}/{len(sourceL)}", end='')
        foldName = destFolder
        fileName = bnce(txtPath,'jpg')
        pathName = f"{os.path.abspath(destFolder)}/{fileName}"
        if defaultAspect:
            height, width = defaultAspect
        else:
            img = cv2.imread(txtPath.replace('.txt','.jpg')) if "classes.txt" not in txtPath else None
            if type(img)!=type(None):
                height, width, _ = img.shape
            else:
                continue
        xml = xml0.replace('folderX',foldName).replace('filenameX',fileName).replace('pathX',pathName).replace('widthX',str(width)).replace('heightX',str(height))
        for yoloLine in open(txtPath).readlines():
            cid, cx, cy, w, h = yoloLine.split(" ")
            xmin, ymin, xmax, ymax = boxAny2Voc("yolo",cx,cy,w,h,width,height) #
            obj = obj0.replace('nameX',classL[int(cid)]).replace('xminX',str(xmin)).replace('yminX',str(ymin)).replace('xmaxX',str(xmax)).replace('ymaxX',str(ymax))
            xml+=obj
        xml+=end0
        with open(f"{destFolder}/{bnce(fileName,'xml')}",'w') as f:
            f.write(xml)

def coco2voc(sourcePath, destFolder):
    """
    Convert labels from coco to voc.

    sourcePath: str. Path of the annotation file *.json.
    destFolder: str. Path of destination.
    """

    global xml0, obj0, end0
    os.makedirs(destFolder, exist_ok=True)
    D      = json.load( open(sourcePath,"r") )
    classD = { catD['id']:catD['name'] for catD in D['categories'] } # classD={0:'dog', 1:'cat'}
    for i,imgD in enumerate(D['images']):
        print(f"\r{i+1}/{len(D['images'])}", end="")
        foldName = destFolder
        fileName = imgD['file_name']
        pathName = f"{os.path.abspath(destFolder)}/{fileName}"
        height   = imgD['height']
        width    = imgD['width']
        xml = xml0.replace('folderX',foldName).replace('filenameX',fileName).replace('pathX',pathName).replace('widthX',str(width)).replace('heightX',str(height))
        for annotD in filter(lambda d:d['image_id']==imgD['id'], D['annotations']):
            cname = classD[ annotD['category_id'] ]
            xmin, ymin, w, h = annotD['bbox']
            xmin, ymin, xmax, ymax = boxAny2Voc("coco",xmin,ymin,w,h) #
            obj = obj0.replace('nameX',cname).replace('xminX',str(xmin)).replace('yminX',str(ymin)).\
                replace('xmaxX',str(xmax)).replace('ymaxX',str(ymax))
            xml+=obj
        xml+=end0
        with open(f"{destFolder}/{fileName.replace('.jpg','.xml')}",'w') as f:
            f.write(xml)

def yolo2coco(sourceFolder, destPath, classL=None, defaultAspect=None):
    """
    Convert labels from yolo to coco.

    sourceFolder: str. Path of source that contains *.jpg and *.txt.
    destFolder: str. Path of destination.
    classL: list[str] or None. list of class names. If 'classes.txt' is in sourceFolder, this arg can be None.
    defaultAspect: (int,int). Can be specified if all the images have same shape, so the box can be compute without reading all images. 
    """

    classL  = classL if classL else [ line.replace('\n','') for line in open(f"{sourceFolder}/classes.txt","r").readlines() ]        
    D = {"images":[], "annotations":[], "categories": []}
    D["categories"] = [ {"supercategory":"none","id":i,"name":className} for i,className in enumerate(classL,0) ] # index start from 0
    sourceL = sorted([ path for path in glob.glob(f"{sourceFolder}/*.txt") if "classes.txt" not in path ])
    boxId   = 0 # annotation.id
    for id,txtPath in enumerate(sourceL):
        print(f"\r{id+1}/{len(sourceL)}", end='')
        txt = open(txtPath,"r").read()
        filename = bnce(txtPath,'jpg')
        if defaultAspect:
            height, width = defaultAspect
        else:
            height, width, _ = cv2.imread( txtPath.replace(".txt",'.jpg') ).shape
        D["images"].append( {"file_name":filename,"height":height,"width":width,"id":id} )
        for yoloLine in open(txtPath).readlines():
            cid, cx, cy, w, h = yoloLine.split(" ")
            xmin, ymin, xmax, ymax = boxAny2Voc("yolo",cx,cy,w,h,width,height)
            xmin, ymin, w, h       = boxVoc2Any("coco",xmin,ymin,xmax,ymax)
            D["annotations"].append( {"area":w*h,"iscrowd":0,"bbox":[xmin,ymin,w,h],"category_id":int(cid),"ignore":0,"segmentation":[],"image_id":id,"id":boxId} )
            boxId+=1
    os.makedirs(os.path.dirname(destPath), exist_ok=True)
    with open(destPath, "w") as f:
        json.dump(D,f)
        
def coco2yolo(sourcePath, destFolder):
    """
    Convert labels from coco to yolo.

    sourcePath: str. Path of the annotation file *.json.
    destFolder: str. Path of destination.
    """

    os.makedirs(destFolder, exist_ok=True)
    D      = json.load( open(sourcePath,"r") )
    with open(f"{destFolder}/classes.txt","w") as f:
    	for catD in D['categories']:
    	    f.write(f"{catD['name']}\n")
    for i,imgD in enumerate(D['images']):
        print(f"\r{i+1}/{len(D['images'])}", end="")
        height, width = imgD['height'], imgD['width']
        f = open(f"{destFolder}/{bnce(imgD['file_name'],'txt')}","w")
        for annotD in filter(lambda d:d['image_id']==imgD['id'], D['annotations']):
            cid = annotD['category_id']
            xmin, ymin, w, h = annotD['bbox']
            xmin, ymin, xmax, ymax = boxAny2Voc("coco",xmin,ymin,w,h)
            cx, cy, w, h           = boxVoc2Any("yolo",xmin,ymin,xmax,ymax,width,height)
            pad = lambda s:str(s)+'0'*(8-len(str(s)))
            f.write(f"{cid} {pad(cx)} {pad(cy)} {pad(w)} {pad(h)}\n")
        f.close()
