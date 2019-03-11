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

    def compute_iou(self, box1, box2):

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

    def forward(self, detection_result, gt):
        #c_x, c_y, w, h , conf, classes(20)
        batch_size = detection_result.shape[0]
        bbox = 5
        num_class = 20
        #Making ground_truth
        grid_size = 13
        gt = list(gt)
        gt_grid = torch.zeros(batch_size, 5 * (5 + num_class), grid_size, grid_size)
        # gt_onefile
        for batch in range(batch_size):
            # print(batch, gt[batch])
            grid = torch.zeros([5 * (5 + num_class), grid_size, grid_size]).cuda()

            gt_one = gt[batch]
            gt_one = gt_one.split(' ')
            # print(gt_one)
            count = 0
            for one_obj in range(0, len(gt_one), 5):
                x, y, w, h, class_id = float(gt_one[0 + one_obj]), float(gt_one[1 + one_obj]), float(gt_one[2 + one_obj]), \
                                       float(gt_one[3 + one_obj]), int(gt_one[4 + one_obj])
                # print(x, y, w, h, class_id, count, grid[(count*25)+4, :, :])

                # GT의 1 grid 에서 객체가 여러 개일 때 grid 채널 이동
                while grid[(count * 25) + 4, int(x * grid_size), int(y * grid_size)] == 1:
                    count += 1


                grid[(count*25)+0, int(x * grid_size), int(y * grid_size)] = x * grid_size - int(x * grid_size)
                grid[(count*25)+1, int(x * grid_size), int(y * grid_size)] = y * grid_size - int(y * grid_size)
                grid[(count*25)+2, int(x * grid_size), int(y * grid_size)] = w
                grid[(count*25)+3, int(x * grid_size), int(y * grid_size)] = h
                grid[(count*25)+4, int(x * grid_size), int(y * grid_size)] = 1
                grid[(count*25)+(bbox + int(class_id)), int(x * grid_size), int(y * grid_size)] = 1
                count = 0

            # print(grid[4,:,:], grid[29,:,:])

            # # 1 grid 내 GT가 여러 개인 경우의 정답값 만들기
            # while grid[(count * 25) + 4, int(x * grid_size), int(y * grid_size)] == 1:
            #     count+=1
            # gt_boxes = torch.cuda.FloatTensor()
            # for gt_one in range(count):
            #     one_gt_box = grid[(gt_one*25):(gt_one*25)+5, :, :].unsqueeze(0)
            #     gt_boxes = torch.cat((one_gt_box, gt_boxes), 0)
            #     print(gt_boxes.shape)

            gt_grid[batch] = grid


        #loss 함수 만들기
        gt_grid = gt_grid.cuda()

        loss_grid = torch.zeros(batch_size, 25, grid_size, grid_size).cuda()

        gt_box = gt_grid[:, :5, :, :]



        # w, h with anchor boxes
        for box in range(0, 5*(bbox+num_class), bbox+num_class):
            detection_result[:, box + 2, :, :] = detection_result[:, box + 2, :, :].clone() * anchors[int(box/25*2)] / grid_size
            detection_result[:, box + 3, :, :] = detection_result[:, box + 3, :, :].clone() * anchors[int(box/25*2)+1] / grid_size

        #calculate IOUs between gt, pred
        ious = []
        best_bbox_index = 0
        max_iou = 0

        loss_grid = torch.zeros(batch_size, 25, grid_size, grid_size).cuda()

        for box in range(0, 5 * (bbox + num_class), bbox + num_class):
            box_one = detection_result[:, box:box+bbox, :, :].clone()
            iou = self.compute_iou(gt_box, box_one).clone()
            confidence = iou.clone() * detection_result[:, box + 4, :, :].clone()
            detection_result[:, box + 4, :, :] = confidence.clone()
            # ious.append(iou)

            if max_iou > 0:
                best_bbox_index = box
                max_iou = iou

        # print(torch.max(ious, dim=0))
        # _b, _x, _y = np.where(ious == torch.max(ious, dim=0)[0])

        for box in range(len(ious)):
            if box == best_bbox_index:
                continue
            _b, _x, _y = np.where(ious[best_bbox_index] >= ious[box])
            if len(_b) != 0:
                loss_grid[_b, 5:, _x, _y] = detection_result[_b, (25*box)+bbox: (25*box)+(bbox+num_class), _x, _y] = detection_result[_b, (25*box)+4, _x, _y].unsqueeze(1).clone() * detection_result[_b, (25*box)+bbox: (25*box)+(bbox+num_class), _x, _y].clone()
                loss_grid[_b, :5, _x, _y] = detection_result[_b, :5, _x, _y].clone()
                # detection_result[_b, (25*box)+5:, _x, _y] = detection_result[_b, (25*box)+4, _x, _y].unsqueeze(1) * detection_result[_b,  (25*box)+5:, _x, _y]
            _b, _x, _y = np.where(ious[best_bbox_index] < ious[box])
            if len(_b) != 0:
                loss_grid[_b, 5:, _x, _y] = detection_result[_b, (25 * box) + bbox: (25 * box) + (bbox + num_class), _x, _y] = detection_result[_b,(25*box)+4, _x, _y].unsqueeze(1).clone() * detection_result[_b, (25 * box) + bbox: (25 * box) + (bbox + num_class), _x, _y].clone()
                loss_grid[_b, :5, _x, _y] = detection_result[_b, :5, _x, _y].clone()

                detection_result[_b, (25*box)+5:, _x, _y] = detection_result[_b, (25*box)+4, _x, _y].unsqueeze(1) * detection_result[_b, (25*box)+5:, _x, _y]

        detection_result = detection_result.cuda()
        lambda_coord = 5.0
        lambda_noobj = 0.5
        obj_loss, no_obj_loss = 0.0, 0.0
        # c_x, c_y, w, h , conf, class0, class1
        # obj가 있는 부분
        one_img_obj, x_obj, y_obj = np.where(gt_grid[:, 4, :, :] == 1)
        # for box in range(0, 125, 25):
        obj_loss += lambda_coord * (mse(loss_grid[one_img_obj, 0, x_obj, y_obj], gt_grid[one_img_obj, 0, x_obj, y_obj]) +
                                    mse(loss_grid[one_img_obj, 1, x_obj, y_obj], gt_grid[one_img_obj, 1, x_obj, y_obj]) +
                                    mse(loss_grid[one_img_obj, 2, x_obj, y_obj], gt_grid[one_img_obj, 2, x_obj, y_obj]) +
                                    mse(loss_grid[one_img_obj, 3, x_obj, y_obj], gt_grid[one_img_obj, 3, x_obj, y_obj])
                                    ) + \
                    mse(detection_result[one_img_obj, 4, x_obj, y_obj], gt_grid[one_img_obj, 4, x_obj, y_obj])

        # obj가 없는 부분
        no_obj = []
        one_img_noobj, x_noobj, y_noobj = np.where(gt_grid[:, 4, :, :] != 1)
        # for box in range(0, 125, 25):
        if len(one_img_noobj) != 0:
            no_obj_loss += lambda_noobj * mse(loss_grid[one_img_noobj, 4, x_noobj, y_noobj], gt_grid[one_img_noobj, 4, x_noobj, y_noobj])

        # probability of Class
        confidence = 0
        one_img_obj, x_obj, y_obj = np.where(gt_grid[:, 4, :, :] == 1)

        # for box in range(0, 125, 25):
        if len(one_img_obj) != 0:
            confidence += mse(loss_grid[one_img_obj, 5:25, x_obj, y_obj], gt_grid[one_img_obj, 5:25, x_obj, y_obj])


        obj_loss = obj_loss.mean()
        no_obj_loss = no_obj_loss.mean()
        confidence = confidence.mean()
        # print(obj_loss, no_obj_loss, confidence)
        loss = obj_loss + no_obj_loss + confidence
        # print(loss)
        return loss, obj_loss, no_obj_loss, confidence