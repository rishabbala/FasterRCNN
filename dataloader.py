import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as tf

import numpy as np

from typing import Tuple
from PIL import Image
import xmltodict


def collate_fn(collate_input: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int, Tensor]:
    """ 
    Collate Function for the DataLoader

    Args:
        collate_input (Tuple[img, bbox, num_boxes]): The input to the collate function that consists of the image from the DataLoader Tensor[N, C, H, W], bounding box/bbox Tensor[L, 4], num_boxes int
    """

    images = []
    bbox = []
    num_boxes = []
    obj_class = []

    for item in collate_input:
        images.append(item[0])
        bbox.append(item[1])
        num_boxes.append(item[2])
        obj_class.append(item[3])
            
    images = torch.stack(images)
    bbox = torch.cat(bbox, dim=0)
    num_boxes = torch.tensor(num_boxes)
    obj_class = torch.cat(obj_class, dim=0)
    
    return images, bbox, num_boxes, obj_class


class VOCDataset(Dataset):
    
    def __init__(self, set_type: str='train', reshaped_img_size: Tuple[int, int]=(200, 200)):
        self.set_type = set_type
        self.reshaped_img_size = reshaped_img_size
        self.transforms = tf.Compose([
            tf.Resize(reshaped_img_size),
            tf.ToTensor(),
            tf.Normalize(mean=[0.4485, 0.4250, 0.3920], std=[0.2668, 0.2634, 0.2766])
        ])
        self.imageset = './data/VOC/VOCdevkit/VOC2007/ImageSets/Layout/'+self.set_type+'.txt'
        
        self.hflip = tf.RandomHorizontalFlip(1)
        self.vflip = tf.RandomVerticalFlip(1)

        self.name2idx = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}       
        
        with open(self.imageset) as f:
            self.image_nums = f.readlines() ## eg '0009377\n' 
    
    def __len__(self):
        return len(self.image_nums)
    
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        
        num_boxes=0
        
        hflag = 0
        vflag = 0
        i_num = self.image_nums[idx][:-1]
        
        img = Image.open('./data/VOC/VOCdevkit/VOC2007/JPEGImages/'+i_num+'.jpg')
        img = self.transforms(img)
        
        # if np.random.binomial(1, 0.5, 1)[0] == 1:
        #     img = self.hflip(img)
        #     hflag = 1
            
        # if np.random.binomial(1, 0.5, 1)[0] == 1:
        #     img = self.vflip(img)
        #     vflag = 1         
        
        with open('./data/VOC/VOCdevkit/VOC2007/Annotations/'+i_num+'.xml', 'r', encoding='utf-8') as file:
            xml = file.read()
        xml = xmltodict.parse(xml)
        
        img_size = torch.tensor([int(xml['annotation']['size']['height']), int(xml['annotation']['size']['width'])])
        bbox = []
        obj_class = []
        
        if type(xml['annotation']['object']).__name__ != 'list':
            xml['annotation']['object'] = [xml['annotation']['object']]
    
        for obj in xml['annotation']['object']:

            obj_class.append(self.name2idx[obj['name']])

            bbox.append([
                float(obj['bndbox']['ymin'])/img_size[0] * self.reshaped_img_size[0],
                float(obj['bndbox']['xmin'])/img_size[1] * self.reshaped_img_size[1],
                float(obj['bndbox']['ymax'])/img_size[0] * self.reshaped_img_size[0],
                float(obj['bndbox']['xmax'])/img_size[1] * self.reshaped_img_size[1]            
            ])
            
            num_boxes += 1
        
            # if hflag == 1:
            #     bbox[-1][2], bbox[-1][0] = self.reshaped_img_size[1]-bbox[-1][0], self.reshaped_img_size[1]-bbox[-1][2]
                
            # if vflag == 1:
            #     bbox[-1][3], bbox[-1][1] = self.reshaped_img_size[0]-bbox[-1][1], self.reshaped_img_size[0]-bbox[-1][3]
        
        bbox = torch.tensor(bbox)
        obj_class = torch.tensor(obj_class)
        
        return img, bbox, num_boxes, obj_class