# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList

DATA_PATH = "/home/tnguyenhu2/alan_project/maskrcnn-benchmark/datasets/food/"
class FOODDataset(Dataset):
    def __init__(
        self, ann_file, root, transforms=None
    ):
        print ("ann_file: ", ann_file)

        self.img_files = []
        self.label_files = []
        self.ids = []

        ann_contents = list(open(ann_file, 'r'))[1:]
        for path in ann_contents:      
            path = path.split(',')[0]
            self.ids.append(path)

            label_path = DATA_PATH + 'annotations/' + path.replace('/', '_').replace('.png', '.txt').replace(
                '.jpg', '.txt').strip()
            image_path = DATA_PATH + 'image/' + path
            # print (image_path)
            # print (label_path)
            if os.path.isfile(label_path):
                self.img_files.append(image_path)
                self.label_files.append(label_path)


        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx % len(self.img_files)].rstrip()
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        label_path = self.label_files[idx % len(self.img_files)].rstrip()
        label_file = open(label_path, 'r')
        for line in label_file:
            line = line.strip().split()
            line = [int(item) for item in line]
            boxes.append(line[:4])
            labels.append(line[4])


        # print ("bbxes: ", len(boxes), len(boxes[0]))
        # print ("labels: ", len(labels))

        labels = torch.tensor(labels)

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        return image, boxlist, idx

    def get_img_info(self, idx):
        img_path = self.img_files[idx % len(self.img_files)].rstrip()
        image = Image.open(img_path).convert("RGB")
        (width, height) = image.size
        return {"height": height, "width": width}
