import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def mse(x, y):
    return (x-y)**2

anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
anchors = np.array(anchors)

class DetectionLoss(nn.Module):

    def compute_ious(self, box1, boxes):
        x1 = torch.max(box1[0], boxes[0, :])
        y1 = torch.max(box1[1], boxes[1, :])
        x2 = torch.min(box1[2], boxes[2, :])
        y2 = torch.min(box1[3], boxes[3, :])
        area_intersection = (x2 - x1 + 1) * (y2 - y1 + 1)

        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (boxes[2, :] - boxes[0, :] + 1) * (boxes[3, :] - boxes[1, :] + 1)
        area_union = area_box1 + area_box2 - area_intersection
        iou = area_intersection / area_union

        return iou
    def forward(self, detection_result, gt):
        #c_x, c_y, w, h , conf, classes(20)
        batch_size = detection_result.shape[0]
        bbox = 5
        num_class = 20
        #Making ground_truth
        grid_size = 13
        gt = list(gt)
        gt_grid = torch.zeros(batch_size, 5 * (5 + num_class), grid_size, grid_size)
        weight_grid = torch.zeros(batch_size, 5 * (5 + num_class), grid_size, grid_size)

        # w, h with anchor boxes
        for box in range(0, 5 * (bbox + num_class), bbox + num_class):
            detection_result[:, box + 2, :, :] = detection_result[:, box + 2, :, :].clone() * anchors[int(box / 25 * 2)] / grid_size
            detection_result[:, box + 3, :, :] = detection_result[:, box + 3, :, :].clone() * anchors[int(box / 25 * 2) + 1] / grid_size

        for batch in range(batch_size):
            gt_one = gt[batch]
            gt_one = gt_one.split(' ')
            # print(gt_one)
            for one_obj in range(0, len(gt_one), 5):
                x, y, w, h, class_id = float(gt_one[0 + one_obj]), float(gt_one[1 + one_obj]), float(gt_one[2 + one_obj]), \
                                       float(gt_one[3 + one_obj]), int(gt_one[4 + one_obj])
                one_obj_box = [x, y, w, h]
                one_obj_box = torch.tensor(one_obj_box).cuda()
                #해당 좌표에 대한 anchor box 만들기
                anchor_boxes = torch.tensor([
                                detection_result[batch, 0::25, int(x * grid_size), int(y * grid_size)].data.cpu().numpy(),
                                detection_result[batch, 1::25, int(x * grid_size), int(y * grid_size)].data.cpu().numpy(),
                                detection_result[batch, 2::25, int(x * grid_size), int(y * grid_size)].data.cpu().numpy(),
                                detection_result[batch, 3::25, int(x * grid_size), int(y * grid_size)].data.cpu().numpy()]).cuda()

                #box1 대한 IOU 계산
                index = np.argmax(self.compute_ious(one_obj_box , anchor_boxes))
                #객체가 있는 곳의 Anchor Box channel에 (x,y,w,h,c, one_hot_class) 입력
                gt_grid[batch, index*25:index*25+5, int(x * grid_size), int(y * grid_size)] = torch.tensor([x, y, w, h, 1])
                one_hot_class = torch.zeros(20)
                one_hot_class[class_id] = 1
                gt_grid[batch, index * 25 + 5:index * 25 + 25, int(x * grid_size), int(y * grid_size)] = one_hot_class
                weight_grid[batch, index * 25:index * 25 + 5, int(x * grid_size), int(y * grid_size)] = torch.tensor([5, 5, 5, 5, 1])
                weight_grid[batch, index * 25 + 5:index * 25 + 25, int(x * grid_size), int(y * grid_size)] = torch.ones(20)
            #객체가 없는 곳은 가중치를 0.5 로 만들기
            _c, _x, _y = np.where(weight_grid[batch, 4::25, :, :] != 1)
            weight_grid[batch, _c*25+4, _x, _y] = 0.5

        #loss 함수 만들기
        gt_grid = gt_grid.cuda()
        weight_grid = weight_grid.cuda()
        loss = mse((weight_grid * gt_grid), detection_result).mean()

        return loss