def IOU(boxA, boxB): # VOC
    (xminA, yminA, xmaxA, ymaxA), (xminB, yminB, xmaxB, ymaxB) = boxA, boxB
    inter = max(0,min(ymaxA,ymaxB)-max(yminA,yminB)) * max(0,min(xmaxA,xmaxB)-max(xminA,xminB))
    areaA = (ymaxA-yminA) * (xmaxA-xminA)
    areaB = (ymaxB-yminB) * (xmaxB-xminB)
    return inter / (areaA+areaB-inter)

def NMS(bboxes, boxesType="yoloFloat", threshold=0.3): # bboxes: np.array
    alive, adopt = set(range(len(bboxes))), []
    while len(alive)>=2:
        ma = min(alive)
        adopt.append( ma )
        boxA = getattr(boxAny2Voc,boxesType)(bboxes[ma][0], bboxes[ma][1], bboxes[ma][2], bboxes[ma][3], width=1000, height=1000)
        alive.remove( ma )
        for idx in alive.copy():
            boxB = getattr(boxAny2Voc,boxesType)(bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3], width=1000, height=1000)
            iou  = IOU(boxA,boxB)
            if iou>=threshold:
                alive.remove(idx)
    if len(alive)==1:
        adopt.append(alive.pop())
    return adopt