import torch
import cv2

def inference(img, ss, models, transform, device):
    ss.setBaseImage(img) 
    ss.switchToSelectiveSearchFast() 
    rects = ss.process()
    
    ROIs = []
    bboxs = []
    
    for rect in rects:
        x, y, w, h = rect
        bb = {'x1': x,
              'y1': y,
              'x2': x+w,
              'y2': y+h} 
        img_croped=img[bb['y1']:bb['y2'],bb['x1']:bb['x2']]
        img_resized = cv2.resize(img_croped, (224, 224))
        bboxs.append(bb)
        ROIs.append(img_resized)
    
    ROImodel, Tumormodel = models

    assert ROImodel.training == False
    assert Tumormodel.training == False
    
    with torch.no_grad():
        transformed_data_list = [transform(data) for data in ROIs]
        batch = torch.stack(transformed_data_list)
        output1 = ROImodel(batch.to(device))
        idx = output1.argmax()
        ROI = ROIs[idx]
        ROI = transform(ROI)
        input_ = ROI.unsqueeze(0).to(device)
        acc = Tumormodel(input_).item() * 100

        if acc >= 50:
            color = (51, 255, 153)
            txt = f'positive {acc:.2f}%'
        else:
            color = (255, 51 ,51)
            txt = f'negative {acc:.2f}%'
        color = (51, 255, 153) if acc >= 50 else (255, 51, 51)
        pt1 = (bboxs[idx]['x1'], bboxs[idx]['y1'])
        pt2 = (bboxs[idx]['x2'], bboxs[idx]['y2'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        cv2.rectangle(img, pt1, pt2, color, 2)
        text_size, _ = cv2.getTextSize(txt, font, font_scale, thickness)
        x, y = pt1
        text_x = x
        text_y = y - text_size[1]
        cv2.putText(img, txt, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return img
