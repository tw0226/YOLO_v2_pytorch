import torch
import model
import losses
from torch import optim
from torch.utils.data import DataLoader
from dataloader import MyDataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from collections import OrderedDict
import xml.etree.ElementTree
from torchvision import transforms

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

def post_processing_boxes(y_pred, test_image, grid_size=13):
    height, width, c = test_image.shape
    img = test_image.copy()
    y_pred = y_pred[0]
    box_pred = None
    for x in range(grid_size):
        for y in range(grid_size):
            for box in range(0, 125, 25):

                y_pred[box + 0, x, y] = int((y_pred[box + 0, x, y] + x) * width / grid_size)
                y_pred[box + 1, x, y] = int((y_pred[box + 1, x, y] + y) * height / grid_size)
                if int(y_pred[box + 2, x, y] * width) > width:
                    y_pred[box + 2, x, y] = width
                else:
                    y_pred[box + 2, x, y] = int(y_pred[box + 2, x, y] * width)
                if int(y_pred[box + 3, x, y] * height) > height:
                    y_pred[box + 3, x, y] = height
                else:
                    y_pred[box + 3, x, y] = int(y_pred[box + 3, x, y] * height)

    for box in range(0, 125, 25):
        if box_pred is None:
            box_pred = y_pred[:25, :, :].unsqueeze(3)
        else:
            box_pred = torch.cat((box_pred, y_pred[box:box+25, :, :].unsqueeze(3)), 3)
    box_pred = box_pred.permute(3, 1, 2, 0)

    box_pred = box_pred.reshape(5*13*13, 25)
    boxes = box_pred[:, :5]
    scores_of_class = box_pred[:, 5:]
    under_threshold = np.where(scores_of_class<0.2)
    scores_of_class[under_threshold] = 0
    scores_of_class = scores_of_class.data.cpu().numpy()
    boxes = boxes.data.cpu().numpy()
    for index in range(20):
        box_list = nms(boxes, scores_of_class[:, index], threshold=0.5)
        false = np.where(box_list == False)
        for box in range(5*13*13):
            if box_list[box] == False:
                continue
            x, y, w, h = boxes[box, 0], boxes[box, 1], boxes[box, 2], boxes[box, 3]
            index = np.argmax(scores_of_class[box, :])
            score = np.max(scores_of_class[box, :])
            if score > 0:
                # print(scores[box, :], category[index], x, y, w, h)

                pt1 = int(x - w / 2), int(y - h / 2)
                pt2 = int(x + w / 2), int(y + h / 2)
                # print(scores_of_class, cls, score)

                img = cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=(colors[index]))
                # print(cls, box_index, max_index, score)
                cv.putText(img, category[index], pt1, cv.FONT_HERSHEY_TRIPLEX, 0.4, color=colors[index])
    return img
def xml_parse(file):
    xml_parse = xml.etree.ElementTree.parse(file+'.xml').getroot()

    category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    img_size = []
    for size in xml_parse.findall('size'):
        width = size.findtext('width')
        height = size.findtext('height')
    # x_axis = width, y_axis = height

    classes = np.zeros(20).astype(int)
    classes = classes.tolist()
    label = []
    for object in xml_parse.findall('object'):
        name = object.findtext('name')
        for bndbox in object.findall('bndbox'):
            xmin = bndbox.findtext('xmin')
            xmax = bndbox.findtext('xmax')
            ymin = bndbox.findtext('ymin')
            ymax = bndbox.findtext('ymax')

        classes[category.index(name)] = 1
        x_center = (int(xmin) + int(xmax)) / (int(width) * 2)
        y_center = (int(ymin) + int(ymax)) / (int(height) * 2)
        x_width = (int(xmax) - int(xmin)) / int(width)
        y_height = (int(ymax) - int(ymin)) / int(height)
        label.append(x_center)
        label.append(y_center)
        label.append(x_width)
        label.append(y_height)
        label.append(category.index(name))
        # label.append([x_center, y_center, x_width, y_height, category.index(name)])
    label = " ".join(repr(e) for e in label)
    return label
def label_to_grid(label):
    grid_size = 13
    num_class = 20
    bbox = 5
    grid = torch.zeros([5 * (5 + num_class), grid_size, grid_size]).cuda()
    count=0
    for one_obj in range(0, len(label), 5):
        x, y, w, h, class_id = float(label[0 + one_obj]), float(label[1 + one_obj]), float(label[2 + one_obj]), \
                               float(label[3 + one_obj]), int(label[4 + one_obj])
        # print(x, y, w, h, class_id, count, grid[(count*25)+4, :, :])

        # GT의 1 grid 에서 객체가 여러 개일 때 grid 채널 이동

        while grid[(count * 25) + 4, int(x * grid_size), int(y * grid_size)] == 1:
            if count == 4:
                break
            count += 1

        grid[(count * 25) + 0, int(x * grid_size), int(y * grid_size)] = x * grid_size - int(x * grid_size)
        grid[(count * 25) + 1, int(x * grid_size), int(y * grid_size)] = y * grid_size - int(y * grid_size)
        grid[(count * 25) + 2, int(x * grid_size), int(y * grid_size)] = w
        grid[(count * 25) + 3, int(x * grid_size), int(y * grid_size)] = h
        grid[(count * 25) + 4, int(x * grid_size), int(y * grid_size)] = 1
        grid[(count * 25) + (bbox + int(class_id)), int(x * grid_size), int(y * grid_size)] = 1
        count = 0
    return grid
def run_demo(training):
    my_model = model.YOLO_v2().cuda()
    state_dict = torch.load('./Weights/YOLO_v2_20.pt')
    my_model.load_state_dict(state_dict)
    my_model.eval()

    grid_size = 7
    criterion = losses.DetectionLoss().cuda()
    test_loss = []
    accuracy = 0
    filepath = 'D:/DATASET/VOC_Dataset/VOC2012_trainval'
    image_name = '2007_000042'
    
    test_image = cv.imread(filepath+'/JPEGImages/'+image_name+'.jpg', cv.IMREAD_COLOR)
    height, width, c = test_image.shape
    label = xml_parse(filepath+'/Annotations/'+image_name)
    label = label.split(' ')

    if training:
        training_epoch = 5000
        my_model.train()
        optimizer = optim.Adam(my_model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = losses.DetectionLoss().cuda()
        epoch_loss = []
    else:
        training_epoch = 1
        my_model.eval()

    for epoch in range(training_epoch):
        img = test_image.copy()
        model_input = cv.resize(test_image, (416, 416))
        model_input = (model_input / 255.).astype(np.float32)
        model_input = torch.from_numpy(model_input)
        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model_input = _normalize(model_input)
        model_input = model_input.permute(2, 0, 1).unsqueeze(0).cuda()

        y_pred = my_model(model_input)
        if training:
            loss, loss1, loss2, conf = criterion(y_pred, label)
            epoch_loss.append(loss.item())
            if epoch % 100 == 0:  # and it > 0:
                print("Step {0}/{1} Loss : {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}".format(epoch, training_epoch,
                                                                 np.mean(epoch_loss), loss1, loss2, conf))
        for one_obj in range(0, len(label), 5):
            x, y, w, h, class_id = float(label[0 + one_obj]), float(label[1 + one_obj]), float(label[2 + one_obj]), \
                                   float(label[3 + one_obj]), int(label[4 + one_obj])
            pt1 = int((x - w / 2) * int(width)), int((y - h / 2) * int(height))
            pt2 = int((x + w / 2) * int(width)), int((y + h / 2) * int(height))
            test_image = cv.rectangle(img=test_image, pt1=pt1, pt2=pt2, color=colors[class_id])
            cv.putText(test_image, category[class_id], pt1, cv.FONT_HERSHEY_TRIPLEX, 0.6, color=colors[class_id])
        y_pred_img = post_processing_boxes(y_pred, img)
        cv.imshow("Ground Truth", test_image)
        cv.imshow("Prediction", y_pred_img)
        if training:
            optimizer.zero_grad()
            loss.backward()
        cv.waitKey(0)

if __name__=="__main__":
    run_demo(training=False)
