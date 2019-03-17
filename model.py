import torch.nn as nn
import torch
import math
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as tv
import cv2
import numpy as np
from torch.autograd import Variable
from darknet import Darknet19

class YOLO_v2(nn.Module):
    def __init__(self):
        self.n_boxes = 5
        self.n_classes = 20
        super(YOLO_v2, self).__init__()

        darknet = Darknet19(pretrained=True).features

        self.feature = darknet
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=(self.n_boxes * (5 + self.n_classes)), kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d((self.n_boxes * (5 + self.n_classes))))
            #nn.LeakyReLU(negative_slope=0.1))


    def forward(self, x):
        x = self.feature(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        for box in range(0, self.n_boxes * (5 + self.n_classes), (5 + self.n_classes)):
            x[:, box:box+2, :, :] = torch.sigmoid(x[:, box:box+2, :, :])#.clone()  # x, y
            x[:, box+2:box+4, :, :] = torch.exp(x[:, box+2:box+4, :, :])#.clone()  # w, h
            x[:, box+4:box+5, :, :] = torch.sigmoid(x[:, box+4:box+5, :, :])#.clone()  # probability of object
            # softmax = nn.Softmax2d().cuda()
            # x[:, box + 5:box + 5 + self.n_classes, :, :] = softmax(x[:, box + 5:box + 5 + self.n_classes, :, :].clone())
            x[:, box + 5:box + 5 + self.n_classes, :, :] = torch.sigmoid(x[:, box + 5:box + 5 + self.n_classes, :, :]) #.clone())  #class

        return x