import cv2
from iou import get_iou

def selective_search_(ss, img, gtbb, iou_threshold, positive):

    positive_sample = []
    negative_sample = []

    up_threshold, down_threshold = iou_threshold

    ss.setBaseImage(img) 
    ss.switchToSelectiveSearchFast() 
    rects = ss.process()

    label = 1 if positive == True else 0

    for rect in rects:
        x, y, w, h = rect
        bb2 = {'x1': x,
               'y1': y,
               'x2': x+w,
               'y2': y+h} 
        iou = get_iou(gtbb, bb2)
        img_croped=img[bb2['y1']:bb2['y2'],bb2['x1']:bb2['x2']]
        img_resized = cv2.resize(img_croped, (224, 224))
        data = [img_resized, label]

        if iou >= up_threshold:
            positive_sample.append(data)
        
        elif iou <= down_threshold:
            negative_sample.append(data)

    return positive_sample, negative_sample
