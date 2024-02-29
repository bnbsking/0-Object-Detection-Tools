# launch from main.py -> engine.py -> detregDownstreamInference.py
import cv2, sys, json, os
packagePath = "/home/jovyan/data-vol-1/detreg/zPackages"
sys.path += [] if packagePath in sys.path else [packagePath]
import visualization as vz

classList   = ["PlasticContainer", "Non-PlasticContainer", "PaperContainer", "Non-PaperContainer"] # "Foil"
gtImgFolder = "data/labv2/test_yolo"
gtAntFolder = "data/labv2/test_yolo"
confThreshold = 0.38
D = json.load(open("data/MSCoco/annotations/instances_val2017.json","r")) # generate data -> D['images'][image_id]['id'] = image_id

def main(outputs, image_id, savePath, crossCategoryNMS=True, savetxt=True, savejpg=True):
    os.makedirs(f"{savePath}/txt" if savetxt else ".", exist_ok=True)
    os.makedirs(f"{savePath}/jpg" if savejpg else ".", exist_ok=True)
    
    prefix = D['images'][image_id]['file_name'].split('.')[0]
    bboxes = outputs['pred_boxes'][0]                   # (300,(cx,cy,w,h)) float
    classScores = outputs['pred_logits'][0].softmax(-1) # (300,classes+1) float        
    topConfIdx = classScores[:,1:].max(axis=1).values.sort(descending=True).indices # get max conf of each query then get their ranking index # (300,) int
    bboxes = bboxes[topConfIdx]                         # (300,(cx,cy,w,h)) float
    scores, classes = classScores[topConfIdx][:,1:].max(axis=1) # (300,) float, (300,) int
    removeNums = {}
            
    if crossCategoryNMS:
        adopt = vz.NMS(bboxes.cpu().numpy(), boxesType="yoloFloat", threshold=0.3)
        removeNums["crossCategoryNMS"] = len(bboxes)-len(adopt)
        bboxes, scores, classes = bboxes[adopt], scores[adopt], classes[adopt]
            
    if savetxt:
        with open(f"{savePath}/txt/{prefix}.txt", "w") as f:
            height, width, _ = cv2.imread(f"{gtImgFolder}/{prefix}.jpg").shape
            for cid, score, (cx,cy,w,h) in zip(classes,scores,bboxes):
                score= round(float(score),4) 
                xmin = int((float(cx)-float(w)/2)*width)
                ymin = int((float(cy)-float(h)/2)*height)
                xmax = int((float(cx)+float(w)/2)*width)
                ymax = int((float(cy)+float(h)/2)*height)
                f.write(f"{cid} {score} {xmin} {ymin} {xmax} {ymax}\n")
            
    if savejpg:
        if confThreshold:
            adopt = list(filter(lambda i:scores[i]>=confThreshold, range(len(bboxes))))
            removeNums["confidenceThreshold"] = len(bboxes)-len(adopt)
            bboxes, scores, classes = bboxes[adopt], scores[adopt], classes[adopt]
        vz.show(f"{gtImgFolder}/{prefix}.jpg", f"{gtAntFolder}/{prefix}.txt", "yoloFloat", bboxes, classes, scores, classList, f"{savePath}/jpg", (1.5,1.5))