# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

border_color = (242, 180, 10)

def draw_boundary(frame, x, y, w, h, border_color):
    # Top left border
    cv2.line(frame, (x, y), (x + (w//5), y), border_color, 2)
    cv2.line(frame, (x, y), (x, y+(h//5)), border_color, 2)

    # Top right border
    cv2.line(frame, (x+((w//5)*4), y), (x+w, y), border_color, 2)
    cv2.line(frame, (x+w, y), (x+w, y+(h//5)), border_color, 2)

    # Bottom left border
    cv2.line(frame, (x, (y+(h//5*4))), (x, y+h), border_color, 2)
    cv2.line(frame, (x, (y+h)), (x + (w//5), y+h), border_color, 2)

    # Bottom right border
    cv2.line(frame, (x+((w//5)*4), y+h), (x + w, y + h), border_color, 2)
    cv2.line(frame, (x+w, (y+(h//5*4))), (x+w, y+h), border_color, 2)


def detect(frame, net, transform):
    # getting width and height of the frame
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    Variable(x.unsqueeze(0))  
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    
    for i in range(detections.size(1)):
        j = 0
        # detections = [batch, number of classes, number of occurence, (score, x0, y0, x1, y1)]
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            draw_boundary(frame, int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]), 
                          border_color)
            # Put the name of the ID
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1]) - 10), cv2.FONT_HERSHEY_DUPLEX, .4,
                border_color, 2, cv2.LINE_AA)
            
            j += 1
            
    return frame
            
