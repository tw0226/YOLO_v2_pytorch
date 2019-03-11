import torch
import model
import losses
import losses_v2
from torch import optim
from torch.utils.data import DataLoader
from dataloader import MyDataset
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torch.autograd import Variable

colors = [np.random.rand(3) * 255 for i in range(20)]
category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def run_train():
    learning_rate=1e-3
    folder_path = "D:/Dataset/VOC_Dataset/"
    train_dataset = MyDataset(folder_path=folder_path+'VOC2012_trainval', train_category='train')
    val_dataset = MyDataset(folder_path=folder_path+'VOC2012_trainval', train_category='val')
    test_dataset = MyDataset(folder_path=folder_path + 'VOC2007_test', train_category='test')
    batch_size = 16
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    my_model = model.YOLO_v2().cuda()
    # my_model.load_state_dict(torch.load('./Weights/YOLO_v1_tiny_min2.pt'))
    my_model.train()
    # my_model = torch.nn.DataParallel(model.YOLO_v1(), device_ids=[0]).cuda()
    # optimizer = optim.SGD(my_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(my_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5)
    training_epoch = 135
    grid_size = 7
    criterion = losses.DetectionLoss().cuda()
    train_loss = []
    val_loss = []

    for epoch in range(training_epoch):
        x_line = [i+1 for i in range(epoch+1)]
        epoch_loss = []
        print('------training------')
        for it, data in enumerate(train_data_loader):
            x = data[0].cuda()
            y = data[1]
            path = data[2][0]
            y_pred = my_model(x).cuda()

            loss, loss1, loss2, conf = criterion(y_pred, y)
            loss.cuda()
            img = cv.imread(path, cv.IMREAD_COLOR)
            height, width, c = img.shape
            gt = [y[0]]
            # print(path)
            line_per_detection = []
            for line in gt:
                line = line.split(' ')
                line_per_detection.append(line)
            for row in line_per_detection:
                # print(row)
                for box in range(0, len(row), 5):
                    x, y, w, h, class_id = float(row[box + 0]), float(row[box + 1]), float(row[box + 2]), float(row[box + 3]), int(row[box + 4])
                    pt1 = int((x - w / 2) * int(width)), int((y - h / 2) * int(height))
                    pt2 = int((x + w / 2) * int(width)), int((y + h / 2) * int(height))
                    cv.putText(img, category[class_id], pt1, cv.FONT_HERSHEY_TRIPLEX, 0.4, color=colors[class_id])
                    img = cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=colors[class_id])

            epoch_loss.append(loss.item())
            if it % 300 == 0 : #and it > 0:
                print("Step {0} : Iteration [{1}/{2}], Loss : {3:0.4f}, {4:0.4f}, {5:0.4f}, {6:0.4f}".format(epoch, it, len(train_data_loader), np.mean(epoch_loss),
                                                                                                             loss1, loss2, conf))
                # pred, pred_img, pred_box = NonMaxSupression(y_pred, path, grid_size)
                # cv.imshow("GroundTruth", img)
                # cv.imshow("Prediction", pred_img)
                cv.waitKey(0)
            optimizer.zero_grad()
            loss.backward()
            scheduler.optimizer.step()

        train_loss.append(np.mean(epoch_loss))
        epoch_loss = []

        print('------validation------')
        for it, data in enumerate(val_data_loader):
            x = data[0].cuda()
            y = data[1]
            path = data[2]
            y_pred = my_model(x)
            loss, loss1, loss2, conf = criterion(y_pred, y)
            loss.cuda()
            epoch_loss.append(loss.item())
            if it % 300 == 0 and it > 0:

                print("Step {0} : Iteration [{1}/{2}], Loss : {3:0.4f}, {4:0.4f}, {5:0.4f}, {6:0.4f}".format(epoch, it, len(train_data_loader), np.mean(epoch_loss),
                                                                                                             loss1, loss2, conf))
                # pred, pred_img = NonMaxSupression(y_pred, path, grid_size)
                # print(pred.shape, pred_img.shape)
                # cv.imshow("pred", pred_img)
                # cv.waitKey(0)

        val_loss.append(np.mean(epoch_loss))
        print("Epoch : {0} / {1} \t Loss : {2:0.4f} {3:0.4f}, {4:0.4f}, {5:0.4f}".format(epoch, training_epoch, np.mean(epoch_loss), loss1, loss2, conf))
        plt.plot(x_line, train_loss, 'r-', label='train')
        plt.plot(x_line, val_loss, 'b-', label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('YOLO_v1_tiny')
        # plt.show()
        plt.savefig('YOLO_v2_loss.png', dpi=300)

        torch.save(my_model.state_dict(), 'Weights/YOLO_v2_tiny_{}.pt'.format(epoch + 1))

if __name__=="__main__":
    run_train()
    # x_line = [i for i in range(10)]
    # y = np.random((10))
    # plt.plot(x_line, y)
    # plt.show()