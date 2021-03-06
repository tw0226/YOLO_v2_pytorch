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

    def compute_ious(self, box1, anchor_boxes, y_offset, x_offset):
        # change grid( center_x, center_y, w, h ) for iou computation
        box1_xmin = box1[0] - box1[2] / 2
        box1_ymin = box1[1] - box1[3] / 2
        box1_xmax = box1[0] + box1[2] / 2
        box1_ymax = box1[1] + box1[3] / 2

        boxes_xmin = anchor_boxes[0]
        boxes_ymin = anchor_boxes[1]
        boxes_width = anchor_boxes[2]
        boxes_height = anchor_boxes[3]

        boxes_xmin = (boxes_xmin + x_offset) / 13 - boxes_width / 2
        boxes_ymin = (boxes_ymin + y_offset) / 13 - boxes_height / 2
        boxes_xmax = boxes_xmin + boxes_width
        boxes_ymax = boxes_ymin + boxes_height

        x1 = torch.max(box1_xmin, boxes_xmin)
        y1 = torch.max(box1_ymin, boxes_ymin)
        x2 = torch.min(box1_xmax, boxes_xmax)
        y2 = torch.min(box1_ymax, boxes_ymax)

        area_intersection = (x2 - x1 + 1) * (y2 - y1 + 1)

        area_box1 = (box1_xmax - box1_xmin + 1) * (box1_ymax - box1_ymin + 1)
        area_box2 = (boxes_xmax - boxes_xmin + 1) * (boxes_ymax - boxes_ymin + 1)
        area_union = area_box1 + area_box2 - area_intersection
        iou = area_intersection / area_union
        
        return iou
    
    def forward(self, model_output, text_label):
        # output = (batch, 125, 13, 13)
        # 125 =  ( c_x, c_y, w, h , conf, classes(20) ) * 5
        batch_size = model_output.shape[0]
        num_anchor_boxes = 5
        grid_size = 13
        num_class = 20

        #Making ground_truth

        text_label = list(text_label)
        label_grid = torch.zeros(batch_size, 5 * (5 + num_class), grid_size, grid_size)
        weight_grid = torch.zeros(batch_size, 5 * (5 + num_class), grid_size, grid_size)

        # w, h with anchor boxes
        for box in range(0, 5 * (num_anchor_boxes + num_class), num_anchor_boxes + num_class):
            model_output[:, box + 2, :, :] = model_output[:, box + 2, :, :] * anchors[int(box / 25 * 2)] / grid_size
            model_output[:, box + 3, :, :] = model_output[:, box + 3, :, :] * anchors[int(box / 25 * 2) + 1] / grid_size

        # loss function
        for batch in range(batch_size):
            gt_boxes = text_label[batch]
            gt_boxes = gt_boxes.split(' ')
            # print(gt_boxes)
            for one_obj in range(0, len(gt_boxes), 5):
                x, y, w, h, class_id = float(gt_boxes[0 + one_obj]), float(gt_boxes[1 + one_obj]), float(gt_boxes[2 + one_obj]), \
                                       float(gt_boxes[3 + one_obj]), int(gt_boxes[4 + one_obj])
                one_obj_box = [x, y, w, h]
                one_obj_box = torch.tensor(one_obj_box)#.cuda()
                # making acnchor box in object grid.
                anchor_boxes = torch.tensor([
                                model_output[batch, 0::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy(),  # x (5)
                                model_output[batch, 1::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy(),  # y (5)
                                model_output[batch, 2::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy(),  # w (5)
                                model_output[batch, 3::25, int(y * grid_size), int(x * grid_size)].data.cpu().numpy()]) # h (5)

                #calculation between iou1 and anchor boxes in object grid
                ious = self.compute_ious(one_obj_box, anchor_boxes, int(y * grid_size), int(x * grid_size))
                box_index = np.argmax(ious)
                # print(ious, box_index)
                #input (x,y,w,h,c, one_hot_of_class) to channel existing object
                label_grid[batch, box_index*25:box_index*25+5, int(y * grid_size), int(x * grid_size)] = \
                    torch.tensor([x*grid_size - int(x * grid_size), y * grid_size - int(y * grid_size), w, h, 1])
                weight_grid[batch, box_index * 25:box_index * 25 + 5, int(y * grid_size), int(x * grid_size)] = \
                    torch.tensor([5, 5, 5, 5, 1])
                one_hot_class = torch.zeros(20)
                one_hot_class[class_id] = 1
                label_grid[batch, box_index * 25 + 5:box_index * 25 + 25, int(y * grid_size), int(x * grid_size)] = one_hot_class
                weight_grid[batch, box_index * 25 + 5:box_index * 25 + 25, int(y * grid_size), int(x * grid_size)] = torch.ones(20)

            #make weight of grid having no object to 0.5
            _c, _y, _x = np.where(label_grid[batch, 4::25, :, :] != 1)
            weight_grid[batch, _c*25+4, _y, _x] = 0.5

        label_grid = label_grid.cuda()
        weight_grid = weight_grid.cuda()

        loss = torch.sum(weight_grid * (label_grid - model_output) * (label_grid - model_output))

        ### check which grid is learning well ###

        #object loss
        obj_b, obj_c, obj_y, obj_x = np.where(label_grid[:, 4::25, :, :] == 1)

        obj_loss = 0
        for box_index in range(5):
            obj_loss += weight_grid[obj_b, obj_c * 25 + box_index, obj_y, obj_x] * \
                        mse(label_grid[obj_b, obj_c * 25 + box_index, obj_y, obj_x],
                            model_output[obj_b, obj_c * 25 + box_index, obj_y, obj_x])

        #no object loss
        no_obj_b, no_obj_c, no_obj_y, no_obj_x = np.where(label_grid[:, 4::25, :, :] != 1)
        no_obj_loss = weight_grid[no_obj_b, no_obj_c * 25 + 4, no_obj_y, no_obj_x] * \
                      mse(label_grid[no_obj_b, no_obj_c * 25 + 4, no_obj_y, no_obj_x],
                          model_output[no_obj_b, no_obj_c * 25 + 4, no_obj_y, no_obj_x])

        #confidence loss
        conf_loss = 0
        conf_b, conf_c, conf_y, conf_x = np.where(label_grid[:, 4::25, :, :] == 1)
        for c_box_index in range(20):
            conf_loss+= weight_grid[conf_b, conf_c*25+5+c_box_index, conf_y, conf_x] * \
                        mse(label_grid[conf_b, conf_c*25+5+c_box_index, conf_y, conf_x],
                            model_output[conf_b, conf_c*25+5+c_box_index, conf_y, conf_x])

        obj_loss = obj_loss.sum()
        no_obj_loss = no_obj_loss.sum()
        conf_loss = conf_loss.sum()
        # print(loss, obj_loss, no_obj_loss, conf_loss)

        return loss, obj_loss, no_obj_loss, conf_loss