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

anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
anchors = np.array(anchors)
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


def compute_iou_box(box1, box2):
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])
    # print(xx2, xx1, yy2, yy1)
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    intersection = w * h
    iou = intersection / (area2 + area1 - intersection)
    return iou

def compute_iou_between_label(label, pred_label):
    correct = 0
    total = len(label) // 5
    for label_index in range(len(label), 5):
        box1 = [label[label_index + 0], label[label_index + 1], label[label_index + 2], label[label_index + 3],
                label[label_index + 4]]
        for pred_index in range(len(pred_label), 5):
            box2 = [pred_label[pred_index + 0], pred_label[pred_index + 1],
                    pred_label[pred_index + 2], pred_label[pred_index + 3], pred_label[pred_index + 4]]

            iou = compute_iou_box(box1, box2)
            if iou > 0.5 and box1[4] == box2[4]:
                correct += 1
    return correct, total

def post_processing_boxes(y_pred, test_image, grid_size=13):
    height, width, c = test_image.shape
    img = test_image.copy()
    label_pred = []

    for batch in range(y_pred.shape[0]):
        y_pred_in_batch = y_pred[batch]
        box_pred = []

        for box in range(0, 125, 25):
            box_pred.append(y_pred_in_batch[box:box+25, :, :].unsqueeze(0))

        box_pred = torch.cat(box_pred, 0)
        box_pred = box_pred.data.cpu().numpy()
        # 5, 13, 13, 25
        # print(box_pred.shape)

        for box in range(5):
            one_label_pred =[]
            for y_grid in range(13):
                for x_grid in range(13):
                    x, y, w, h, c = box_pred[box, :5, y_grid, x_grid]



                    class_prob = c * box_pred[box, 5:, y_grid, x_grid]
                    under_score = np.where(class_prob < 0.3)[0]
                    if len(under_score) != 0:
                        class_prob[under_score] = 0

                    # print(x + x_grid / 13, (x+x_grid)/13)
                    x = (x + x_grid) / 13
                    y = (y + y_grid) / 13

                    w = w * anchors[box * 2] / 13
                    h = h * anchors[box * 2 + 1] / 13
                    # print(x, y)
                    max_index = np.argmax(class_prob)
                    score = np.max(class_prob)
                    if score > 0.9:
                        # print(x_grid, y_grid, box_pred[box, 5+max_index, y_grid, x_grid], category[max_index])
                        # print(class_prob)
                        pt1 = int((x - w / 2) * int(width)), int((y - h / 2) * int(height))
                        pt2 = int((x + w / 2) * int(width)), int((y + h / 2) * int(height))
                        c_p = int(x * width), int(y * height)
                        img = cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=(colors[max_index]), thickness=2)
                        img = cv.circle(img=img, center=c_p, radius=3, color=colors[max_index], thickness=-1)
                        cv.putText(img, category[max_index] + '{0:0.4f}'.format(class_prob[max_index]), pt1, cv.FONT_HERSHEY_TRIPLEX, 0.4, color=colors[max_index])
                        one_label_pred.append([x, y, w, h, max_index])

            label_pred.append(one_label_pred)

    return img, label_pred


def xml_parse(file):
    xml_parse = xml.etree.ElementTree.parse(file + '.xml').getroot()

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

def run_demo(training):
    my_model = model.YOLO_v2().cuda()
    # state_dict = torch.load('./Weights/YOLO_v2_200.pt')
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # my_model.load_state_dict(new_state_dict)
    # my_model.eval()

    grid_size = 13
    criterion = losses.DetectionLoss().cuda()
    test_loss = []
    accuracy = 0
    filepath = 'D:/DATASET/VOC_Dataset/VOC2012_trainval'
    image_name = '2007_000042'
    train_loss = []

    test_image = cv.imread(filepath + '/JPEGImages/' + image_name + '.jpg', cv.IMREAD_COLOR)
    height, width, c = test_image.shape
    label = xml_parse(filepath + '/Annotations/' + image_name)
    list_label = [label]
    label = label.split(' ')
    _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if training:
        training_epoch = 100001
        my_model.train()
        optimizer = optim.Adam(my_model.parameters(), lr=1e-3)#, weight_decay=1e-5)
        criterion = losses.DetectionLoss().cuda()
        epoch_loss = []
    else:
        training_epoch = 1

    img = test_image.copy()
    for epoch in range(training_epoch):
        x_line = [i + 1 for i in range(epoch + 1)]
        my_model.train()
        model_input = cv.resize(test_image, (416, 416))
        model_input = (model_input / 255.).astype(np.float32)
        model_input = torch.from_numpy(model_input)
        model_input = _normalize(model_input)
        model_input = model_input.permute(2, 0, 1).unsqueeze(0).cuda()
        y_pred = my_model(model_input)
        #y_pred_clone = y_pred.clone()
        for one_obj in range(0, len(label), 5):
            x, y, w, h, class_id = float(label[0 + one_obj]), float(label[1 + one_obj]), float(label[2 + one_obj]), \
                                   float(label[3 + one_obj]), int(label[4 + one_obj])
            pt1 = int((x - w / 2) * int(width)), int((y - h / 2) * int(height))
            pt2 = int((x + w / 2) * int(width)), int((y + h / 2) * int(height))
            c_p = int(x * width), int(y * height)
            cv.circle(test_image, center=c_p, radius=3, color=colors[class_id], thickness=-1)
            test_image = cv.rectangle(img=test_image, pt1=pt1, pt2=pt2, color=colors[class_id], thickness=2)
            cv.putText(test_image, category[class_id], pt1, cv.FONT_HERSHEY_TRIPLEX, 0.6, color=colors[class_id])

        # print(img.shape, y_pred_img.shape, y_pred_label)
        # correct, total = compute_iou_between_label(label, y_pred_label)
        if epoch % 50 == 0 and epoch > 100:
            y_pred_img, y_pred_label = post_processing_boxes(y_pred.clone(), img.copy())
            cv.imshow("Prediction", y_pred_img)
            cv.imshow("Ground Truth", test_image)
            cv.waitKey(33)
            if epoch % 1000 == 0:
                cv.imwrite('{0}_{1}.jpg'.format(image_name, epoch), y_pred_img)
        if training:
            # loss, loss1, loss2, conf = criterion(y_pred, list_label)
            loss, obj_loss, no_obj_loss, conf_loss = criterion(y_pred, list_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            train_loss.append(np.mean(epoch_loss))
            if epoch % 50 == 0:  # and it > 0:
                print("Step {0}/{1} Loss : {2:0.4f} {3:0.4f} {4:0.4f} {5:0.4f}".format(
                    epoch, training_epoch, np.mean(epoch_loss), obj_loss, no_obj_loss, conf_loss))
                plt.plot(x_line, train_loss, 'r-', label='train')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('YOLO_v2')
                plt.savefig('YOLO_v2_loss_one_img.png', dpi=300)
                # print("Step {0}/{1} Loss : {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f} Accuracy : {6:0.2f} {7}/{8} ".format(epoch, training_epoch,
                #                                                  np.mean(epoch_loss), loss1, loss2, conf, correct/total, correct, total))
            epoch_loss=[]
            # if epoch % 1000:
            #     torch.save(my_model.state_dict(), 'test/YOLO_v2_{}.pt'.format(epoch+1))
        else:
            continue
            # cv.imshow("Ground Truth", test_image)
            # cv.imshow("Prediction", y_pred_img)
            # cv.waitKey(0)


if __name__ == "__main__":
    run_demo(training=True)
