"""
Code adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

Creates a class that allows for processing of images and masks by a MaskRCNN
"""

import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import random
import cv2
import json
import numpy as np


class CloudsDataset(torch.utils.data.Dataset):


    def __init__(self, params, root, split, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.file_list = os.listdir(root + "images")
        self.SEED = 2023
        random.Random(self.SEED).shuffle(self.file_list)
        #We want train to have 2/3 of the train data, val to have 1/3 of the train data, and test to have all of the test data
        if split == "train":
            self.split = "train"
            self.file_list = self.file_list[:int(0.66 * len(self.file_list))]
        if split == "val":
            self.split = "train"
            self.file_list = self.file_list[int(0.66 * len(self.file_list)):]
        if split == "test":
            self.split = "test"
            self.file_list = self.file_list
                
    def __getitem__(self, idx):
        #Read in image and convert from json to mask
        height, width = 512, 512
        img = read_image(self.root + "images/" + self.file_list[idx])
        mask = json_to_mask(self.file_list[idx][:-4], self.split, height, width)

        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = mask.shape[0]

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(mask)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(mask)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.file_list)

def json_to_mask(json_name, split, height, width):
    f = open("../Dataset/"+split+"/labels/"+json_name+".json")
    im = cv2.imread("../Dataset/"+split+"/images/"+json_name+".png")
    annotation = json.load(f)
    masks = np.zeros((len(annotation["shapes"]), 512, 512))
    #if there are no clouds in the image give an empty mask
    # if len(annotation["shapes"]) == 0:
    #     masks = np.zeros((1, 512, 512))
    for i in range(len(annotation["shapes"])):

        #Make sure that the labels are scaled correctly with the images and masks
        if im.shape[0] == 512:
            points = [[int(point[0]), int(point[1])] for point in annotation["shapes"][i]["points"]]
        if im.shape[0] == 1024:
            points = [[int(point[0] / 2), int(point[1] / 2)] for point in annotation["shapes"][i]["points"]]

        points = np.array(points)
        cv2.fillPoly(masks[i, :, :], pts=[points], color=((i + 1) * 10, 0, 0))
        masks[i, :, :] = (masks[i, :, :] == (i + 1) * 10)
    #print("Returning masks of shape", masks.shape, "with max", np.max(masks))
    return torch.from_numpy(masks).to(dtype=torch.float32)