import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def mse(x, y):
    return (x - y) ** 2


anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
anchors = np.array(anchors)
anchors = anchors * 416 / 32

bbox = 5
num_class = 20
grid_size = 13
class DetectionLoss(nn.Module):
    def compute_iou_batch(self, box1, box2):

        x1 = torch.max(box1[:, 0, :, :], box2[:, 0, :, :])
        y1 = torch.max(box1[:, 1, :, :], box2[:, 1, :, :])
        x2 = torch.min(box1[:, 2, :, :], box2[:, 2, :, :])
        y2 = torch.min(box1[:, 3, :, :], box2[:, 3, :, :])
        area_intersection = (x2 - x1 + 1) * (y2 - y1 + 1)
        area_box1 = (box1[:, 2, :, :] - box1[:, 0, :, :] + 1) * (box1[:, 3, :, :] - box1[:, 1, :, :] + 1)
        area_box2 = (box2[:, 2, :, :] - box2[:, 0, :, :] + 1) * (box2[:, 3, :, :] - box2[:, 1, :, :] + 1)
        area_union = area_box1 + area_box2 - area_intersection
        iou = area_intersection / area_union
        return iou

    def compute_iou(self, box1, box2):

        x1 = torch.max(box1[0, :, :], box2[0, :, :])
        y1 = torch.max(box1[1, :, :], box2[1, :, :])
        x2 = torch.min(box1[2, :, :], box2[2, :, :])
        y2 = torch.min(box1[3, :, :], box2[3, :, :])
        area_intersection = (x2 - x1 + 1) * (y2 - y1 + 1)
        area_box1 = (box1[2, :, :] - box1[0, :, :] + 1) * (box1[3, :, :] - box1[1, :, :] + 1)
        area_box2 = (box2[2, :, :] - box2[0, :, :] + 1) * (box2[3, :, :] - box2[1, :, :] + 1)
        area_union = area_box1 + area_box2 - area_intersection
        iou = area_intersection / area_union
        return iou

    def conver_label_to_grid(self, gt_one):
        grid = torch.zeros([5 * (5 + num_class), grid_size, grid_size]).cuda()
        count = 0
        for one_obj in range(0, len(gt_one), 5):
            x, y, w, h, class_id = float(gt_one[0 + one_obj]), float(gt_one[1 + one_obj]), float(
                gt_one[2 + one_obj]), \
                                   float(gt_one[3 + one_obj]), int(gt_one[4 + one_obj])
            # print(x, y, w, h, class_id, count, grid[(count*25)+4, :, :])

            # GT의 1 grid 에서 객체가 여러 개일 때 grid 채널 이동
            while grid[(count * 25) + 4, int(x * grid_size), int(y * grid_size)] == 1:
                count += 1

            grid[(count * 25) + 0, int(x * grid_size), int(y * grid_size)] = x * grid_size - int(x * grid_size)
            grid[(count * 25) + 1, int(x * grid_size), int(y * grid_size)] = y * grid_size - int(y * grid_size)
            grid[(count * 25) + 2, int(x * grid_size), int(y * grid_size)] = w
            grid[(count * 25) + 3, int(x * grid_size), int(y * grid_size)] = h
            grid[(count * 25) + 4, int(x * grid_size), int(y * grid_size)] = 1
            grid[(count * 25) + (bbox + int(class_id)), int(x * grid_size), int(y * grid_size)] = 1
            count = 0
        return grid
    def forward(self, detection_result, gt):
        # c_x, c_y, w, h , conf, classes(20)
        batch_size = detection_result.shape[0]
        bbox = 5
        num_class = 20
        # Making ground_truth
        gt = list(gt)
        gt_grid = torch.zeros(batch_size, 5 * (5 + num_class), grid_size, grid_size)
        # gt_onefile
        for batch in range(batch_size):
            # print(batch, gt[batch])
            # grid = torch.zeros([5 * (5 + num_class), grid_size, grid_size]).cuda()

            gt_one = gt[batch]
            gt_one = gt_one.split(' ')
            # print(gt_one)


            #label to grid converting
            grid = self.conver_label_to_grid(gt_one)
            gt_boxes = torch.cuda.FloatTensor()

            # for gt_one in range(count):
            #     one_gt_box = grid[(gt_one * 25):(gt_one * 25) + 5, :, :].unsqueeze(0)
            #     gt_boxes = torch.cat((one_gt_box, gt_boxes), 0)
            #     print(gt_boxes.shape)


            #미리 주어진 anchor box 곱하기

            prior_w = torch.cuda.FloatTensor(anchors[::2]) / grid_size
            prior_h = torch.cuda.FloatTensor(anchors[1::2]) / grid_size

            prior_w = torch.einsum('ab,c->cab', (detection_result[batch, 2, :, :], prior_w))   #  (13,13) *(5) -> 5,13,13)
            prior_h = torch.einsum('ab,c->cab', (detection_result[batch, 3, :, :], prior_h))   #  (13,13) *(5) -> 5,13,13)

            max_iou = torch.zeros(grid_size, grid_size)
            for index in range(bbox):  # each anchor_box
                box1 = [detection_result[batch, 0, :, :], detection_result[batch, 1, :, :], prior_w[index, :, :], prior_h[index, :, :]]
                gt_box = [grid[0, :, :], grid[1, :, :], grid[2, :, :], grid[3, :, :]]
                iou = self.compute_iou(box1, gt_box)
                _x, _y = np.where(iou >= max_iou)
                if len(_x) > 0:
                    detection_result[batch, index * 25 + 4, :, :] = iou[_x, _y] * detection_result[batch, index * 25 + 4, _x, _y]

                _x, _y = np.where(iou < max_iou)
                if len(_x) > 0:
                    detection_result[batch, index * 25 + 4, :, :] = max_iou[_x, _y] * detection_result[batch, index * 25 + 4, _x, _y]



            gt_grid[batch] = grid

            # loss 함수 만들기



            # calculate IOUs between gt, pred
            for box in range(bbox):
                _x, _y = np.where(grid[box*25+4, :, :] > 0)
                if len(_x) != 0 and box > 0:
                    continue
            for index, gt_one_box in enumerate(gt_boxes):
                ious = []
                best_bbox_index = 0
                max_iou = 0

                for box in range(0, 5 * (bbox + num_class), bbox + num_class):
                    box_one = detection_result[batch, box:box + bbox, :, :]
                    iou = self.compute_iou(gt_one_box, box_one)
                    print(iou)
                    ious.append(iou)
                    if max_iou > 0:
                        best_bbox_index = box
                    max_iou = iou

                # Confidence = Pr(object)*IOU(gt_boxes[index], Pred[Highest_index])
                detection_result[batch, (25 * index) + 4, :, :] = detection_result[batch, 25 * best_bbox_index + 4, :, :] * ious[box]

                for box in range(len(ious)):
                    if box == best_bbox_index:
                        continue
                    _x, _y = np.where(ious[best_bbox_index] >= ious[box])
                    if len(_x) != 0:


                        detection_result[_b, (25 * box) + bbox: (25 * box) + (bbox + num_class), _x, _y] = \
                            detection_result[batch, (25 * box) + 4, _x, _y].unsqueeze(1) \
                            * detection_result[batch, (25 * box) + bbox: (25 * box) + (bbox + num_class), _x, _y]

                    _x, _y = np.where(ious[best_bbox_index] < ious[box])
                    if len(_x) != 0:
                        detection_result[batch, (25 * box) + bbox: (25 * box) + (bbox + num_class), _x, _y] = \
                            detection_result[batch, (25 * box) + 4, _x, _y].unsqueeze(1) \
                            * detection_result[batch, (25 * box) + bbox: (25 * box) + (bbox + num_class), _x, _y]


        detection_result = detection_result.cuda()
        lambda_coord = 5.0
        lambda_noobj = 0.5
        obj_loss, no_obj_loss = 0.0, 0.0
        # c_x, c_y, w, h , conf, class0, class1
        # obj가 있는 부분
        one_img_obj, c_obj, x_obj, y_obj = np.where(gt_grid[:, 4::25, :, :] == 1)

        if len(one_img_obj) != 0:
            obj_loss += lambda_coord * (mse(detection_result[one_img_obj, c_obj * 25 + 0, x_obj, y_obj],
                                            gt_grid[one_img_obj, c_obj * 25 + 0, x_obj, y_obj]).mean() +
                                        mse(detection_result[one_img_obj, c_obj * 25 + 1, x_obj, y_obj],
                                            gt_grid[one_img_obj, c_obj * 25 + 1, x_obj, y_obj]).mean() +
                                        mse(detection_result[one_img_obj, c_obj * 25 + 2, x_obj, y_obj],
                                            gt_grid[one_img_obj, c_obj * 25 + 2, x_obj, y_obj]).mean() +
                                        mse(detection_result[one_img_obj, c_obj * 25 + 3, x_obj, y_obj],
                                            gt_grid[one_img_obj, c_obj * 25 + 3, x_obj, y_obj]).mean()
                                        ) + \
                        mse(detection_result[one_img_obj, c_obj * 25 + 4, x_obj, y_obj],
                            gt_grid[one_img_obj, c_obj * 25 + 4, x_obj, y_obj]).mean()


        # obj가 없는 부분
        no_obj = []
        one_img_noobj, c_noobj, x_noobj, y_noobj = np.where(gt_grid[:, 4::25, :, :] != 1)
        if len(one_img_noobj) != 0:
            no_obj_loss += lambda_noobj * mse(detection_result[one_img_noobj, c_noobj * 25 + 4, x_noobj, y_noobj],
                                              gt_grid[one_img_noobj, c_noobj * 25 + 4, x_noobj, y_noobj]).mean()

        # probability of Class
        confidence = 0
        # one_img_obj, c_obj, x_obj, y_obj = np.where(gt_grid[:, 4::25, :, :] == 1)

        for box in range(0, 5 * (bbox + num_class), bbox + num_class):
            one_img_obj, x_obj, y_obj = np.where(gt_grid[:, box + 4, :, :] == 1)
            if len(one_img_obj) != 0:
                confidence += mse(detection_result[one_img_obj, box + 5:box + 25, x_obj, y_obj],
                                  gt_grid[one_img_obj, box + 5:box + 25, x_obj, y_obj]).mean()
        # obj_loss = obj_loss.sum()

        print(obj_loss, no_obj_loss, confidence)
        loss = obj_loss + no_obj_loss + confidence
        print(loss)
        return loss, obj_loss, no_obj_loss, confidence