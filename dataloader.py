import numpy as np
import cv2
import torch
import xml.etree.ElementTree
from torchvision import transforms
class MyDataset:
    def __init__(self, folder_path, train_category):
        self.folder_path = folder_path
        file = open(folder_path+'/ImageSets/Main/'+train_category+'.txt', 'r')
        filelist=[]
        for line in file.readlines():
            line = line.strip()
            filelist.append(line)
        self.list = filelist
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        # idx = self.list.index('000121')  # 000121
        # idx = 2
        img = cv2.imread(self.folder_path + '/JPEGImages/' + self.list[idx] + ".jpg", cv2.IMREAD_COLOR)
        path = self.folder_path + '/JPEGImages/' + self.list[idx] + ".jpg"

        origin_img = img
        img = cv2.resize(img, (416, 416))
        img = (img / 255.).astype(np.float32)
        img = torch.from_numpy(img)
        img = self._normalize(img)
        img = img.permute(2, 0, 1)

        label = self.folder_path + '/Annotations/' + self.list[idx] + ".xml"

        xml_parse = xml.etree.ElementTree.parse(label).getroot()
        category = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        for size in xml_parse.findall('size'):
            width = size.findtext('width')
            height = size.findtext('height')


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
        label = " ".join(repr(e) for e in label)

        return img, label, path

    def __len__(self):
        return len(self.list)

