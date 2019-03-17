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

    def compute_ious(self, box1, boxes, off_y, off_x):
        new_box = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]

        new_boxes = torch.zeros(boxes.shape)
        new_boxes[0] = (boxes[0] + off_x) / 13 - boxes[2]/2
        new_boxes[1] = (boxes[1] + off_y) / 13 - boxes[3]/2
        new_boxes[2] = new_boxes[0] + boxes[2]
        new_boxes[3] = new_boxes[1] + boxes[3]

        x1 = torch.max(new_box[0], new_boxes[0, :])
        y1 = torch.max(new_box[1], new_boxes[1, :])
        x2 = torch.min(new_box[2], new_boxes[2, :])
        y2 = torch.min(new_box[3], new_boxes[3, :])

        area_intersection = (x2 - x1 + 1) * (y2 - y1 + 1)

        area_box1 = (new_box[2] - new_box[0] + 1) * (new_box[3] - new_box[1] + 1)
        area_box2 = (new_boxes[2, :] - new_boxes[0, :] + 1) * (new_boxes[3, :] - new_boxes[1, :] + 1)
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
            detection_result[:, box + 2, :, :] = detection_result[:, box + 2, :, :] * anchors[int(box / 25 * 2)] / grid_size
            detection_result[:, box + 3, :, :] = detection_result[:, box + 3, :, :] * anchors[int(box / 25 * 2) + 1] / grid_size
        for batch in range(batch_size):
            gt_one = gt[batch]
            gt_one = gt_one.split(' ')
            # print(gt_one)
            for one_obj in range(0, len(gt_one), 5):
                x, y, w, h, class_id = float(gt_one[0 + one_obj]), float(gt_one[1 + one_obj]), float(gt_one[2 + one_obj]), \
                                       float(gt_one[3 + one_obj]), int(gt_one[4 + one_obj])
                one_obj_box = [x, y, w, h]
                one_obj_box = torch.tensor(one_obj_box)#.cuda()
                #해당 좌표에 대한 anchor box 만들기
                anchor_boxes = torch.tensor([
                                detection_result[batch, 0::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy(),
                                detection_result[batch, 1::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy(),
                                detection_result[batch, 2::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy(),
                                detection_result[batch, 3::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy()])#.cuda()

                #box1 대한 IOU 계산

                ious = self.compute_ious(one_obj_box, anchor_boxes, int(y * grid_size), int(x * grid_size))

                index = np.argmax(ious)

                # for box in range(5):
                #     detection_result[batch, box*25+4, int(y * grid_size), int(x * grid_size)] = \
                #         detection_result[batch, box*25+4, int(y * grid_size), int(x * grid_size)].clone() * ious[box].cuda()

                #객체가 있는 곳의 Anchor Box channel에 (x,y,w,h,c, one_hot_class) 입력
                gt_grid[batch, index*25:index*25+5, int(y * grid_size), int(x * grid_size)] = \
                    torch.tensor([x*grid_size - int(x * grid_size), y * grid_size - int(y * grid_size), w, h, 1])
                weight_grid[batch, index * 25:index * 25 + 5, int(y * grid_size), int(x * grid_size)] = \
                    torch.tensor([5, 5, 5, 5, 1])
                one_hot_class = torch.zeros(20)
                one_hot_class[class_id] = 1
                gt_grid[batch, index * 25 + 5:index * 25 + 25, int(y * grid_size), int(x * grid_size)] = one_hot_class
                weight_grid[batch, index * 25 + 5:index * 25 + 25, int(y * grid_size), int(x * grid_size)] = torch.ones(20)

            #객체가 없는 곳은 가중치를 0.5 로 만들기
            _c, _y, _x = np.where(gt_grid[batch, 4::25, :, :] != 1)
            weight_grid[batch, _c*25+4, _y, _x] = 0.5


        #loss 함수 만들기
        gt_grid = gt_grid.cuda()
        weight_grid = weight_grid.cuda()

        loss = torch.sum(weight_grid * (gt_grid- detection_result)* (gt_grid - detection_result))

        # # 어느 부분이 학습되고 있는지 확인용
        obj_b, obj_c, obj_y, obj_x = np.where(gt_grid[:, 4::25, :, :] == 1)

        obj_loss=0
        for index in range(5):
            obj_loss += weight_grid[obj_b, obj_c * 25 + index, obj_y, obj_x] * \
                        mse(gt_grid[obj_b, obj_c * 25 + index, obj_y, obj_x],
                            detection_result[obj_b, obj_c * 25 + index, obj_y, obj_x])


        no_obj_b, no_obj_c, no_obj_y, no_obj_x = np.where(gt_grid[:, 4::25, :, :] != 1)
        no_obj_loss = weight_grid[no_obj_b, no_obj_c * 25 + 4, no_obj_y, no_obj_x] * \
                      mse(gt_grid[no_obj_b, no_obj_c * 25 + 4, no_obj_y, no_obj_x],
                          detection_result[no_obj_b, no_obj_c * 25 + 4, no_obj_y, no_obj_x])
        conf_loss = 0
        conf_b, conf_c, conf_y, conf_x = np.where(gt_grid[:, 4::25, :, :] == 1)
        for c_index in range(20):
            conf_loss+= weight_grid[conf_b, conf_c*25+5+c_index, conf_y, conf_x] * \
                        mse(gt_grid[conf_b, conf_c*25+5+c_index, conf_y, conf_x],
                            detection_result[conf_b, conf_c*25+5+c_index, conf_y, conf_x])
        obj_loss = obj_loss.sum()
        no_obj_loss = no_obj_loss.sum()
        conf_loss = conf_loss.sum()
        # loss = obj_loss + no_obj_loss +conf_loss
        # print(loss, obj_loss, no_obj_loss, conf_loss)

        return loss, obj_loss, no_obj_loss, conf_loss