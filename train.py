import torch
import model
import losses
from torch import optim
from torch.utils.data import DataLoader
from dataloader import MyDataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import OrderedDict
colors = [np.random.rand(3) * 255 for i in range(20)]
category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
anchors = np.array(anchors)

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
    # np.seterr(divide='ignore', invalid='ignore')
    iou = intersection / (areas + area - intersection)
    return iou

def post_processing_boxes(y_pred, test_image, grid_size=13):
    batch_size, height, width, c = test_image.shape
    label_pred = []
    return_img = torch.zeros(test_image.shape)
    return_img = return_img.permute(0,2,3,1)

    img = test_image.clone()
    img = img.permute(0, 2, 3, 1)
    img = img.data.cpu().numpy()
    y_pred = y_pred.data.cpu().numpy()
    box_pred = None
    for x in range(grid_size):
        for y in range(grid_size):
            for box in range(0, 125, 25):
                y_pred[:, box + 0, x, y] = ((y_pred[:, box + 0, x, y] + x) * width / grid_size).astype(int)
                y_pred[:, box + 1, x, y] = ((y_pred[:, box + 1, x, y] + x) * height / grid_size).astype(int)
                y_pred[:, box + 2, x, y] = (y_pred[:, box + 2, x, y] * anchors[int(box/25*2)] * width).astype(int)
                y_pred[:, box + 3, x, y] = (y_pred[:, box + 3, x, y] * anchors[int(box/25*2)+1] * height).astype(int)

    for box in range(0, 125, 25):
        if box_pred is None:
            box_pred = np.expand_dims(y_pred[:, :25, :, :], axis=4)
            # box_pred = y_pred[:, :25, :, :].unsqueeze(3)
        else:
            y_pred_concat = np.expand_dims(y_pred[:, :25, :, :], axis=4)
            box_pred = np.concatenate((box_pred, y_pred_concat), axis=4)
            # box_pred = torch.cat((box_pred, y_pred[:, box:box+25, :, :].unsqueeze(3)), 3)
    box_pred = np.transpose(box_pred, (0, 4, 1, 2, 3))
    # box_pred = box_pred.permute(3, 1, 2, 0)
    box_pred = box_pred.reshape(-1, 5*13*13, 25)

    boxes = box_pred[:, :, :5]
    scores_of_class = box_pred[:, :, 5:]
    under_threshold = np.where(scores_of_class<0.3)
    scores_of_class[under_threshold] = 0

    for batch in range(batch_size):
        one_label_pred = []
        for index in range(20):
            box_list = nms(boxes[batch], scores_of_class[batch, :, index], threshold=0.5)
            false = np.where(box_list == False)
            for box in range(5*13*13):
                if box_list[box] == False:
                    continue
                x, y, w, h = boxes[batch, box, 0], boxes[batch, box, 1], boxes[batch, box, 2], boxes[batch, box, 3]
                max_index = np.argmax(scores_of_class[batch, box, :])
                score = np.max(scores_of_class[batch, box, :])
                if score > 0:
                    one_label_pred.append([x,y,w,h, max_index])
                    # print(category[index], score)
                    # print(scores[box, :], category[index], x, y, w, h)

                    pt1 = int(x - w / 2), int(y - h / 2)
                    pt2 = int(x + w / 2), int(y + h / 2)
                    # print(scores_of_class, cls, score)
                    img = cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=(colors[max_index]))
                    # print(cls, box_index, max_index, score)
                    cv.putText(img, category[max_index], pt1, cv.FONT_HERSHEY_TRIPLEX, 0.4, color=colors[max_index])
        return_img = torch.from_numpy(img)
        label_pred.append(one_label_pred)
    return return_img, label_pred

def compute_iou_between_label(label, pred_label):
    correct=0
    total = len(label)
    for label_index in range(len(label)):
        box1 = [label[label_index][0], label[label_index][1], label[label_index][2], label[label_index][3], label[label_index][4]]
        iou = compute_iou(pred_label, box1)
        indices = np.where(iou>0.5)
        # print(index)
        for index in indices:
            if box1[4] == pred_label[index, 4]:
                correct+=1
    return correct, total

def run_train():
    learning_rate=1e-3
    folder_path = "D:/Dataset/VOC_Dataset/"
    train_dataset = MyDataset(folder_path=folder_path+'VOC2012_trainval', train_category='train')
    val_dataset = MyDataset(folder_path=folder_path+'VOC2012_trainval', train_category='val')
    batch_size = 8
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    my_model = model.YOLO_v2().cuda()
    state_dict = torch.load('./Weights/YOLO_v2_40.pt')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    my_model.load_state_dict(new_state_dict)
    my_model.eval()
    my_model.train()
    # my_model = torch.nn.DataParallel(model.YOLO_v1(), device_ids=[0]).cuda()
    # optimizer = optim.SGD(my_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(my_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5)
    training_epoch = 135
    grid_size = 7
    criterion = losses.DetectionLoss().cuda()
    train_loss = []
    val_loss = []
    sum_correct, sum= 0, 0
    for epoch in range(training_epoch):
        x_line = [i+1 for i in range(epoch+1)]
        epoch_loss = []
        print('------training------')
        for it, data in enumerate(train_data_loader):
            x = data[0].cuda()
            y = data[1]
            path = data[2][0]
            y_pred = my_model(x).cuda()
            gt = list(y)
            loss = criterion(y_pred, y)
            loss = loss.cuda()


            img = x.clone()
            # y_pred_img, y_pred_label = post_processing_boxes(y_pred, img)
            # for batch in range(batch_size):
            #     gt_batch_one = np.asarray(gt[batch].split(' '), dtype=np.float32)
            #     pred_batch_one = np.asarray(y_pred_label[batch], dtype=np.float32)
            #     gt_batch_one = gt_batch_one.reshape(-1,5)
            #     correct, total = compute_iou_between_label(gt_batch_one, pred_batch_one)
            #     sum_correct+=correct
            #     sum+=total
            # img = cv.imread(path, cv.IMREAD_COLOR)
            # height, width, c = img.shape
            # gt = [y[0]]
            # # print(path)
            # line_per_detection = []
            # for line in gt:
            #     line = line.split(' ')
            #     line_per_detection.append(line)
            # for row in line_per_detection:
            #     # print(row)
            #     for box in range(0, len(row), 5):
            #         x, y, w, h, class_id = float(row[box + 0]), float(row[box + 1]), float(row[box + 2]), float(row[box + 3]), int(row[box + 4])
            #         pt1 = int((x - w / 2) * int(width)), int((y - h / 2) * int(height))
            #         pt2 = int((x + w / 2) * int(width)), int((y + h / 2) * int(height))
            #         cv.putText(img, category[class_id], pt1, cv.FONT_HERSHEY_TRIPLEX, 0.4, color=colors[class_id])
            #         img = cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=colors[class_id])

            epoch_loss.append(loss.item())
            if it % 100 == 0 : #and it > 0:
                print("Step {0}/{1} Loss : {2:0.4f}".format(it, len(train_data_loader), np.mean(epoch_loss)))
                # print("Step {0}/{1} Loss : {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f} Accuracy : {6:0.2f} {7}/{8} ".format(it, len(train_data_loader),
                #             np.mean(epoch_loss), loss1, loss2, conf, sum_correct / sum, sum_correct, sum))
                # pred, pred_img, pred_box = NonMaxSupression(y_pred, path, grid_size)
                # cv.imshow("GroundTruth", img)
                # cv.imshow("Prediction", pred_img)
                # cv.waitKey(0)
            optimizer.zero_grad()
            loss.backward()
            scheduler.optimizer.step()

        train_loss.append(np.mean(epoch_loss))
        epoch_loss = []

        print('------validation------')
        for it, data in enumerate(val_data_loader):
            x = data[0].cuda()
            img = x.clone()
            y = data[1]
            path = data[2]
            y_pred = my_model(x)
            loss, loss1, loss2, conf = criterion(y_pred, y)
            loss.cuda()
            epoch_loss.append(loss.item())
            y_pred_img, y_pred_label = post_processing_boxes(y_pred, img)


            if it % 300 == 0 and it > 0:

                print("Step {0} : Iteration [{1}/{2}], Loss : {3:0.4f}, {4:0.4f}, {5:0.4f}, {6:0.4f}".format(epoch+1, it, len(train_data_loader), np.mean(epoch_loss),
                                                                                                             loss1, loss2, conf))
                # pred, pred_img = NonMaxSupression(y_pred, path, grid_size)
                # print(pred.shape, pred_img.shape)
                # cv.imshow("pred", pred_img)
                # cv.waitKey(0)

        val_loss.append(np.mean(epoch_loss))
        print("Epoch : {0} / {1} \t Loss : {2:0.4f} {3:0.4f}, {4:0.4f}, {5:0.4f}".format(epoch+1, training_epoch, np.mean(epoch_loss), loss1, loss2, conf))
        plt.plot(x_line, train_loss, 'r-', label='train')
        plt.plot(x_line, val_loss, 'b-', label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('YOLO_v2')
        # plt.show()
        plt.savefig('YOLO_v2_loss.png', dpi=300)

        torch.save(my_model.state_dict(), 'Weights/YOLO_v2_{}.pt'.format(epoch + 1))

if __name__=="__main__":
    run_train()