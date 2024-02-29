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
    colors = [ (1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1) ] # 
else:
    colors = set()
    while len(colors)<90:
        colors.add( tuple([round(random.random(),2) for i in range(3)]) )
    colors = list(colors)


class BoxAny2Voc:
    def voc(xmin,ymin,xmax,ymax,width=None,height=None):
        return int(xmin),int(ymin), int(xmax), int(ymax)
    
    def yolo(cx,cy,w,h,width=None,height=None): # width, height only valid
        xmin = int((float(cx)-float(w)/2)*float(width))
        ymin = int((float(cy)-float(h)/2)*float(height))
        xmax = int((float(cx)+float(w)/2)*float(width))
        ymax = int((float(cy)+float(h)/2)*float(height))
        return xmin, ymin, xmax, ymax
    
    def yolo_int(cx,cy,w,h,width=None,height=None):
        xmin = int(int(cx)-int(w)/2)
        ymin = int(int(cy)-int(h)/2)
        xmax = int(int(cx)+int(w)/2)
        ymax = int(int(cy)+int(h)/2)
        return xmin, ymin, xmax, ymax
    
    def coco(xmin,ymin,w,h,width=None,height=None):
        return int(xmin), int(ymin), int(xmin)+int(w), int(ymin)+int(h)
    

class CocoCache:
    """
    Non-self class, attributes are shared everywhere
    This Cache cannot be refreshed
    """
    D, cache = {}, False
    def load(path):
        if not CocoCache.cache or not CocoCache.D:
            print("fill cache")
            CocoCache.D = json.load(open(path, 'r'))
        return CocoCache.D


def get_annotation(img_path, ant_path, class_list):
    """
    return boxes (N,4), cids (N,) 
    """
    # Pascal VOC
    if ".xml" in ant_path: 
        xml = open(ant_path,"r").read()
        name_list = re.findall("<name>(.*)</name>", xml)
        xmin_list = re.findall("<xmin>(.*)</xmin>", xml)
        ymin_list = re.findall("<ymin>(.*)</ymin>", xml)
        xmax_list = re.findall("<xmax>(.*)</xmax>", xml)
        ymax_list = re.findall("<ymax>(.*)</ymax>", xml)
        boxes, cids = [], []
        for name, xmin, ymin, xmax, ymax in zip(name_list, xmin_list, ymin_list, xmax_list, ymax_list):
            cids.append( class_list.index(name) )
            xmin, ymin, xmax, ymax = BoxAny2Voc.voc(xmin, ymin, xmax, ymax)
            boxes.append( [xmin, ymin, xmax, ymax] )
        return boxes, cids

    # YOLO
    elif ".txt" in ant_path:
        height, width, _ = cv2.imread(img_path).shape
        boxes, cids = [], []
        for line in open(ant_path,"r").readlines():
            cid, cx, cy, w, h = line.split(" ")
            cids.append( int(cid) )
            if "." in cx:
                xmin, ymin, xmax, ymax = BoxAny2Voc.yolo(cx, cy, w, h, width, height)
            else:
                xmin, ymin, xmax, ymax = BoxAny2Voc.yolo_int(cx, cy, w, h)
            boxes.append( [xmin, ymin, xmax, ymax] )
        return boxes, cids
    
    # COCO
    elif ".json" in ant_path:
        D = CocoCache.load(ant_path)
        img_name = img_path.split('/')[-1]
        id = next( (dic['id'] for dic in D['images'] if dic['file_name']==img_name) )
        ant_dict_list = [ dic for dic in D['annotations'] if dic['image_id']==id ]
        boxes, cids = [], []
        for dic in ant_dict_list:
            cid = dic['category_id']-0 #-1 # cid start from 0 or 1
            cids.append( cid )
            xmin, ymin, w, h = dic['bbox']
            xmin, ymin, xmax, ymax = BoxAny2Voc.coco(xmin, ymin, w, h) 
            boxes.append( [xmin, ymin ,xmax, ymax] )
        return boxes, cids
    
    else:
        raise ValueError("Unkown extension of annotation file")

def show(class_list, img_path, ant_path="", pd_boxes_type="", pd_boxes=None, pd_cids=None, pd_cfs=None, \
    save_path="", box_width=4, value_ratios=(1,1)): # use help(show) for more details
    """
    This function helps visualize an image with varies format of label or prediction.
    
    + class_list: list[str]. List of class names
    + img_path: str. Path to the image (e.g. *.jpg, ... etc.)
    + ant_path: str or None. Path to the annotation (e.g. *.voc, *.txt, *.json). If None, show black img only.
    
    + pd_boxes_type: str. '' or 'voc' or 'yolo' or 'yolo_int' or 'coco'. If None, show black only, (pd_boxes, pd_cids, pd_cfs) are not used.
    + pd_boxes: None or ndarray in shape (N,4)
    + pd_cids: None or ndarray in shape (N,) class index
    + pd_cfs: None or ndarray in shape (N,) confidence
    
    + save_folder: str or None. Save at the folder with same filename. If None, show only but not save.
    + value_ratios: int. Predicted boxes width.
    """
    
    img_raw = cv2.imread(img_path)[:,:,::-1]/255

    # generate img_gt
    if not ant_path:
        img_gt = np.zeros((img_raw.shape[0], img_raw.shape[1], 3))
    else:
        img_gt = img_raw.copy()
        boxes_gt, cids_gt = get_annotation(img_path, ant_path, class_list)
        for (xmin, ymin, xmax, ymax), cid in zip(boxes_gt, cids_gt):
            img_gt[ymin-box_width:ymin+box_width, xmin:xmax, :] = colors[cid]
            img_gt[ymax-box_width:ymax+box_width, xmin:xmax, :] = colors[cid]
            img_gt[ymin:ymax, xmin-box_width:xmin+box_width, :] = colors[cid]
            img_gt[ymin:ymax, xmax-box_width:xmax+box_width, :] = colors[cid]
    
    # generate img_pd
    if not pd_boxes_type:
        img_pd = np.zeros((img_raw.shape[0], img_raw.shape[1], 3))
    else:
        img_pd = img_raw.copy()
        height, width, _ = img_pd.shape
        for i, (b1, b2, b3, b4) in reversed(list(enumerate(pd_boxes))): # plot least conf first
            xmin, ymin, xmax, ymax = getattr(BoxAny2Voc, pd_boxes_type)(b1, b2, b3, b4, width, height)
            img_pd[ymin-box_width:ymin+box_width, xmin:xmax, :] = colors[pd_cids[i]]
            img_pd[ymax-box_width:ymax+box_width, xmin:xmax, :] = colors[pd_cids[i]]
            img_pd[ymin:ymax, xmin-box_width:xmin+box_width, :] = colors[pd_cids[i]]
            img_pd[ymin:ymax, xmax-box_width:xmax+box_width, :] = colors[pd_cids[i]]
            # confidence patches
            ud, td = int(pd_cfs[i]*10), int(pd_cfs[i]*100)%10
            P = getPatch(ud,td,color=colors[pd_cids[i]])
            (ph, pw, _), (rh,rw) = P.shape, value_ratios
            P = cv2.resize( P, (int(pw*rw),int(ph*rh)) )
            try:
                if ymin>=P.shape[0] and xmin+P.shape[1]<img_pd.shape[1]: # upper bar - up
                    img_pd[ymin-P.shape[0]:ymin,xmin:xmin+P.shape[1],:] = P
                elif ymax+P.shape[0]<img_pd.shape[0] and xmin+P.shape[1]<img_pd.shape[1]: # down bar - down
                    img_pd[ymax:ymax+P.shape[0],xmin:xmin+P.shape[1],:] = P
                elif ymin+P.shape[0]<img_pd.shape[0] and xmin+P.shape[1]<img_pd.shape[1]:
                    img_pd[ymin:ymin+P.shape[0],xmin:xmin+P.shape[1],:] = P # upper bar - down
                elif ymax+P.shape[0]>0 and xmin+P.shape[1]<img_pd.shape[1]: # down bar - up
                    img_pd[ymax-P.shape[0]:ymax,xmin:xmin+P.shape[1],:] = P
            except:
                pass

    # plot
    fig = plt.figure(figsize=(20,10))
    fig.set_facecolor("white")

    plt.subplot(1,2,1)
    plt.title("GT", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    for r, g, b in colors:
        c2hex = lambda c: hex(int(c*255))[2:].zfill(2)
        plt.scatter([0], [0], c=f"#{c2hex(r)}{c2hex(g)}{c2hex(b)}")

    plt.legend(labels=class_list, fontsize=16)
    plt.imshow(img_gt)
    
    plt.subplot(1,2,2)
    plt.title("Pred", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.imshow(img_pd)
    
    plt.savefig( f"{save_path}/{img_path.split('/')[-1]}" ) if save_path else plt.show()
    plt.close()