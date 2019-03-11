import torch
import model
import losses
from torch import optim
from torch.utils.data import DataLoader
from dataloader import MyDataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

colors = [np.random.rand(3) * 255 for i in range(20)]
category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def nms(boxes, probs, threshold):
    """Non-Maximum supression.
    Args:
      boxes: array of [cx, cy, w, h] (center format)
      probs: array of probabilities
      threshold: two boxes are considered overlapping if their IOU is largher than
          this threshold
      form: 'center' or 'diagonal'
    Returns:
      keep: array of True or False.
    """

    order = probs.argsort()[::-1]

    keep = [True] * len(order)

    for i in range(len(order) - 1):
        ovps = compute_iou(boxes[order[i + 1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j + i + 1]] = False
    return keep

def compute_iou(boxes, box):
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    # print(xx2, xx1, yy2, yy1)
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    intersection = w * h
    iou = intersection / (areas + area - intersection)
    return iou

def NonMaxSupression(y_pred, path, grid_size=13, nms_conf=0.2):

    batch_size = y_pred.shape[0]
    pred_box = []
    score_list = []
    cls_list = []
    for batch in range(batch_size):
        img = cv.imread(path, cv.IMREAD_COLOR)
        height, width, c = img.shape
        test = y_pred[batch]


        anchor_box = 5

        prediction = torch.zeros(anchor_box, grid_size, grid_size, 25)  # x,y,w,h,conf, conf*class
        for box in range(anchor_box):
            # prediction[box, :, :, :] = test[:, :, 4].unsqueeze(2) * test[:, :, 10:]
            prediction[box, :, :, :5] = test[:, :, box*5:(box+1)*5]
            prediction[box, :, :, 5:] = test[:, :, 4].unsqueeze(2) * test[:, :, 10:]  # confidence * class_probability
            for x in range(grid_size):
                for y in range(grid_size):
                    prediction[box, x, y, 0] = int((prediction[box, x, y, 0] + x) * width / grid_size)
                    prediction[box, x, y, 1] = int((prediction[box, x, y, 1] + y) * height / grid_size)
                    prediction[box, x, y, 2] = int(prediction[box, x, y, 2] * width)
                    prediction[box, x, y, 3] = int(prediction[box, x, y, 3] * height)

        pred = prediction.view(grid_size * grid_size * anchor_box, 25)  # 845 * 25
        # pred = pred.permute(1, 0)  # 25 * 98
        pred = pred.data.cpu().numpy()
        iou_threshold = 0.5

        print(pred.shape)

        boxes = pred[:, :5]
        scores = pred[:, 5:]
        # print(scores.shape)

        box_none, class_none = np.where(scores < nms_conf)
        for box_index, class_index in zip(box_none, class_none):
            scores[box_index][class_index] = 0

        pred = torch.from_numpy(pred).cuda()

        # for index in range(20):
        #     box_list = nms(boxes, scores[index], threshold=0.5)
        #     # print(box_list)
        for index in range(20):
            box_list = nms(boxes, scores[:, index], threshold=0.5)
            # false = np.where(box_list == False)
            for box in range(98):
                if box_list[box] == False:
                    continue
                x, y, w, h = boxes[box, 0], boxes[box, 1], boxes[box, 2], boxes[box, 3]
                index = np.argmax(scores[box, :])
                score = np.max(scores[box, :])
                if score > 0:
                    # print(scores[box, :], category[index], x, y, w, h)

                    pt1 = int(x - w / 2), int(y - h / 2)
                    pt2 = int(x + w / 2), int(y + h / 2)
                    # print(class_probability, cls, score)

                    img = cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=(colors[index]))
                    # print(cls, box_index, max_index, score)
                    cv.putText(img, category[index], pt1, cv.FONT_HERSHEY_TRIPLEX, 0.4, color=colors[index])


        # pred = pred.permute(1, 0)  # 98 * 25
        # for score, cls in zip(score_list, cls_list):
        #     print(score, category[cls])
        # print(count)
        pred = pred.view(7 * 7 * 2, 25)  # 98 * 25 -> 7 * 7 * 25
        pred = pred.view(2, 7, 7, 25)
        # print(count)
        return pred, img, pred_box
