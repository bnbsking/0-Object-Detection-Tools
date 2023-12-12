import os, glob, re, json
import cv2 # yolo2others only

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
" # folderX,filenameX,pathX,widthX,heightX need to be replaced
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
" # nameX,xmin,ymin,xmax,ymax need to be replaced
    end0="</annotation>"
    
def boxAny2Voc(srcType, b1, b2, b3, b4, width=None, height=None):
    if srcType=="voc": # b1,b2,b3,b4 = xmin,ymin,xmax,ymax
        xmin, ymin, xmax, ymax = int(b1), int(b2), int(b3), int(b4)
    elif srcType=="yoloFloat": # b1,b2,b3,b4 = cx,cy,w,h
        xmin = int((float(b1)-float(b3)/2)*float(width))
        ymin = int((float(b2)-float(b4)/2)*float(height))
        xmax = int((float(b1)+float(b3)/2)*float(width))
        ymax = int((float(b2)+float(b4)/2)*float(height))
    elif srcType=="yoloInt": # b1,b2,b3,b4 = cx,cy,w,h
        xmin = int(int(b1)-int(b3)/2)
        ymin = int(int(b2)-int(b4)/2)
        xmax = int(int(b1)+int(b3)/2)
        ymax = int(int(b2)+int(b4)/2)
    elif srcType=="coco": # b1,b2,b3,b4 = xmin,ymin,w,h
        xmin, ymin, xmax, ymax = int(b1), int(b2), int(b1)+int(b3), int(b2)+int(b4)
    else:
        raise KeyError(f"{srcType} Not found")
    return xmin, ymin, xmax, ymax

def boxVoc2Any(desType, xmin, ymin, xmax, ymax, width=None, height=None):
    if desType=="voc":
        return int(xmin), int(ymin), int(xmax), int(ymax)
    elif desType=="yoloFloat":
        cx = round((int(xmin)+int(xmax))/2/float(width),6)
        cy = round((int(ymin)+int(ymax))/2/float(height),6)
        w  = round((int(xmax)-int(xmin))/float(width),6)
        h  = round((int(ymax)-int(ymin))/float(height),6)
        return cx, cy, w, h
    elif desType=="yoloInt":
        cx = int((int(xmin)+int(xmax))/2)
        cy = int((int(ymin)+int(ymax))/2)
        w  = int((int(xmax)-int(xmin)))
        h  = int((int(ymax)-int(ymin)))
        return cx, cy, w, h
    elif desType=="coco":
        xmin = int(xmin)
        ymin = int(ymin)
        w    = int(xmax)-int(xmin)
        h    = int(ymax)-int(ymin)
        return xmin, ymin, w, h
    else:
        raise KeyError(f"{desType} Not found")
        
# yolo means yoloFloat
def voc2yolo(sourceFolder, destFolder, classList):
    with open(f"{destFolder}/classes.txt","w") as f:
        for c in classList:
            f.writelines(f"{c}\n")
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
        with open(f"{destFolder}/{xmlPath.split('/')[-1].replace('.xml','.txt')}", "w") as f:
            for name,xmin,ymin,xmax,ymax in zip(nameL,xminL,yminL,xmaxL,ymaxL):
                id = classList.index(name)
                cx, cy, w, h = boxVoc2Any("yoloFloat",xmin,ymin,xmax,ymax,width,height) #
                pad = lambda s:str(s)+'0'*(8-len(str(s)))
                f.writelines(f"{id} {pad(cx)} {pad(cy)} {pad(w)} {pad(h)}\n")

def voc2coco(sourceFolder, destPath, classList):
    D = {"images":[], "annotations":[], "categories": []}
    D["categories"] = [ {"supercategory":"none","id":i,"name":className} for i,className in enumerate(classList,1) ]
    sourceL = sorted(glob.glob(f"{sourceFolder}/*.xml"))
    boxId = 0 # annotation.id
    for id,xmlPath in enumerate(sourceL):
        print(f"\r{id+1}/{len(sourceL)}", end='')
        xml = open(xmlPath,"r").read()
        filename = xmlPath.split('/')[-1].replace('.xml','.jpg')
        height = int(re.findall("<height>([0-9]*)</height>",xml)[0])
        width  = int(re.findall("<width>([0-9]*)</width>",xml)[0])
        nameL  = re.findall("<name>(.*)</name>",xml)
        xminL  = re.findall("<xmin>(.*)</xmin>",xml)
        yminL  = re.findall("<ymin>(.*)</ymin>",xml)
        xmaxL  = re.findall("<xmax>(.*)</xmax>",xml)
        ymaxL  = re.findall("<ymax>(.*)</ymax>",xml)
        D["images"].append( {"file_name":filename,"height":height,"width":width,"id":id} )
        for i,(name,xmin,ymin,xmax,ymax) in enumerate(zip(nameL,xminL,yminL,xmaxL,ymaxL),1):
            xmin, ymin, w, h = boxVoc2Any("coco",xmin,ymin,xmax,ymax) #
            catid= classList.index(name)+1
            D["annotations"].append( {"area":w*h,"iscrowd":0,"bbox":[xmin,ymin,w,h],"category_id":catid,"ignore":0,"segmentation":[],"image_id":id,"id":boxId} )
            boxId+=1
    with open(destPath, "w") as f:
        json.dump(D,f)
                
def yolo2voc(sourceFolder, destFolder):
    global xml0, obj0, end0
    with open(f"{sourceFolder}/classes.txt","r") as f:
        D = {str(i):key[:-1] for i,key in enumerate(f.readlines())} # e.g. D={0:'dog',1:'cat'}
    sourceL = glob.glob(f"{sourceFolder}/*.txt")
    for i,txtPath in enumerate(sourceL):
        print(f"\r{i+1}/{len(sourceL)}", end='')
        folder   = destFolder.split('/')[-1] if "/" in destFolder else os.getcwd().split('/')[-1]
        filename = txtPath.split('/')[-1].replace('.txt','.jpg')
        path     = f"{os.path.abspath(destFolder)}/{filename}"
        img = cv2.imread(txtPath.replace('.txt','.jpg'))
        if type(img)!=type(None):
            height, width, _ = cv2.imread(txtPath.replace('.txt','.jpg')).shape
        else:
            continue
        xml = xml0.replace('folderX',folder).replace('filenameX',filename).replace('pathX',path).\
            replace('widthX',str(width)).replace('heightX',str(height))
        for yoloLine in open(txtPath).readlines():
            id, cx, cy, w, h = yoloLine.split(" ")
            xmin, ymin, xmax, ymax = boxAny2Voc("yoloFloat",cx,cy,w,h,width,height) #
            obj = obj0.replace('nameX',D[id]).replace('xminX',str(xmin)).replace('yminX',str(ymin)).\
                replace('xmaxX',str(xmax)).replace('ymaxX',str(ymax))
            xml+=obj
        xml+=end0
        with open(f"{destFolder}/{filename.replace('.jpg','.xml')}",'w') as f:
            f.write(xml)
        
def coco2voc(sourceFolder, destFolder):
    global xml0, obj0, end0
    D = json.load( open(f"{sourceFolder}/labels.json","r") )
    for i,imgD in enumerate(D['images']):
        print(f"\r{i}/{len(D['images'])}", end="")
        folder   = destFolder.split('/')[-1] if "/" in destFolder else os.getcwd().split('/')[-1]
        filename = imgD['file_name']
        path     = f"{os.path.abspath(destFolder)}/{filename}"
        height   = imgD['height']
        width    = imgD['width']
        xml = xml0.replace('folderX',folder).replace('filenameX',filename).replace('pathX',path).\
            replace('widthX',str(width)).replace('heightX',str(height))
        classDict = { catD['id']:catD['name'] for catD in D['categories'] } # classDict={0:'dog', 1:'cat'}
        for annotD in filter(lambda d:d['image_id']==imgD['id'], D['annotations']):
            cname = classDict[ annotD['category_id'] ]
            xmin, ymin, w, h = annotD['bbox']
            xmin, ymin, xmax, ymax = boxAny2Voc("coco",xmin,ymin,w,h) #
            obj = obj0.replace('nameX',cname).replace('xminX',str(xmin)).replace('yminX',str(ymin)).\
                replace('xmaxX',str(xmax)).replace('ymaxX',str(ymax))
            xml+=obj
        xml+=end0
        with open(f"{destFolder}/{filename.replace('.jpg','.xml')}",'w') as f:
            f.write(xml)