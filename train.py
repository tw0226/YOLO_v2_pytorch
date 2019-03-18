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

def run_train():
    learning_rate=1e-3
    folder_path = "D:/Dataset/VOC_Dataset/"
    train_dataset = MyDataset(folder_path=folder_path+'VOC2012_trainval', train_category='train')
    val_dataset = MyDataset(folder_path=folder_path+'VOC2012_trainval', train_category='val')
    batch_size = 8
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    my_model = model.YOLO_v2().cuda()
    my_model.train()

    optimizer = optim.Adam(my_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    training_epoch = 135
    grid_size = 13
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
            loss, obj_loss, no_obj_loss, conf_loss = criterion(y_pred, y)
            epoch_loss.append(loss.item())
            if it % 100 == 0 : #and it > 0:
                print(
                    'step [{0:} / {1}] \t loss : {2:3.4f} \t obj_loss : {3:3.4f} \t no_obj_loss : {4:3.4f} \t conf_loss : {5:3.4f}'
                    .format(it, len(train_data_loader), loss, obj_loss, no_obj_loss, conf_loss))
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
            loss, obj_loss, no_obj_loss, conf_loss = criterion(y_pred, y)
            loss.cuda()
            epoch_loss.append(loss.item())

            if it % 300 == 0 and it > 0:
                print(
                    'step [{0:} / {1}] \t loss : {2:3.4f} \t obj_loss : {3:3.4f} \t no_obj_loss : {4:3.4f} \t conf_loss : {5:3.4f}'
                        .format(it, len(val_data_loader), loss, obj_loss, no_obj_loss, conf_loss))
                # pred, pred_img = NonMaxSupression(y_pred, path, grid_size)
                # print(pred.shape, pred_img.shape)
                # cv.imshow("pred", pred_img)
                # cv.waitKey(0)

        val_loss.append(np.mean(epoch_loss))
        print(
            'step [{0:} / {1}] \t loss : {2:3.4f} \t obj_loss : {3:3.4f} \t no_obj_loss : {4:3.4f} \t conf_loss : {5:3.4f}'
            .format(epoch, training_epoch, loss, obj_loss, no_obj_loss, conf_loss))
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