import torch
import torch.nn as nn
import numpy as np

import torchvision
import torchvision.transforms as tf

from typing import Tuple, List
from torch import Tensor


def IOU(boxes1: List, boxes2: List) -> float:
    ## shape (N, 4) where N is num of boxes
    
    area = lambda box: (box[:, 2]-box[:, 0]) * (box[:, 3]-box[:, 1])
    
    upper_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    lower_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_coord = (lower_right - upper_left).clamp(min=0)
    inter_area = inter_coord[:, :, 0] * inter_coord[:, :, 1]
    union_area = area(boxes1)[:, None] + area(boxes2) - inter_area
    
    return inter_area.to(torch.float32)/union_area.to(torch.float32)

def gen_coord2yxhw(boxes: Tensor) -> Tensor:

    assert boxes.shape[-1] == 4
    init_shape = boxes.shape

    boxes = boxes.view(-1, 4)
    boxes = torch.stack(((boxes[:, 2]+boxes[:, 0])/2, (boxes[:, 3]+boxes[:, 1])/2, boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]), dim=1)

    boxes = boxes.view(init_shape)
    return boxes


class GenerateAnchorBoxes():
    '''
    Generates anchor boxes in the original image scale
    '''
    
    def __init__(self, aspect_ratio: List=[0.5, 1, 2], size: List=[0.25, 0.5, 0.75], img_shape: Tuple[int, int]=(200, 200), device: str='cpu'):
        self.aspect_ratio = torch.tensor(aspect_ratio) ## w_bbox/h_bbox
        self.size = torch.tensor(size) ##sqrt(a_bbox/a_img)
        self.img_shape = torch.tensor(img_shape)
        self.img_area = self.img_shape[0] * self.img_shape[1]
        self.box_types = len(self.aspect_ratio) * len(self.size)
        self.num_proposals_per_image = 16
        self.anchors = None
        self.device = device

        
    def generate_centered_anchors(self, aspect_ratio: List, size: List) -> Tensor:
        ## box in image coordinates, centered at (0, 0)
        width = size * torch.sqrt(aspect_ratio) * torch.sqrt(self.img_area)
        height = size * torch.sqrt(self.img_area) / torch.sqrt(aspect_ratio)
        
        return torch.tensor([-height/2, -width/2, height/2, width/2])
    

    def shift_bbox(self, y: int, x: int, bbox:List[Tensor]) -> List[Tensor]:
        shifted_bbox = []
        
        for box in bbox:
            s_box = torch.tensor([box[0]+y, box[1]+x, box[2]+y, box[3]+x])            
            shifted_bbox.append(s_box)
        return shifted_bbox
            
        
    def generate_anchors_from_fmap(self, fmap_shape: Tuple[int, int], train_boxes: Tensor, train_num_boxes: Tensor, train_classes: Tensor) -> Tuple[Tensor, Tensor]:

        if self.anchors == None:
            bbox = []
            
            ## all bboxes in image coordinates generated at (0, 0)
            aspect_ratio, size = torch.meshgrid(self.aspect_ratio, self.size)
            for ar, sz in zip(aspect_ratio.flatten(), size.flatten()):
                bbox.append(self.generate_centered_anchors(ar, sz).to(torch.int32))
                                                
            ## have to check if width or height comes first in torch image
            h_scale = torch.arange(0, self.img_shape[0], torch.floor(self.img_shape[0]/fmap_shape[0]))
            w_scale = torch.arange(0, self.img_shape[1], torch.floor(self.img_shape[1]/fmap_shape[1]))
                    
            self.anchors = []
            
            h_scale, w_scale = torch.meshgrid(h_scale, w_scale)
            for y, x in zip(h_scale.flatten(), w_scale.flatten()):            
                s_bbox = self.shift_bbox(y, x, bbox)
                self.anchors.extend(s_bbox)
            
            self.anchors = torch.stack(self.anchors).to(self.device)
        
        return self.create_training_anchor_class(train_boxes, train_num_boxes, train_classes)


    def create_training_anchor_class(self, train_boxes: Tensor, train_num_boxes: Tensor, train_classes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        prev_index = 0

        anchor_proposals_batch = []
        anchor_class_batch = []
        anchor_pos_batch = []
        anchor_gt_batch = []
        anchor_gt_class_batch = []

        for i in range(train_num_boxes.shape[0]):
            iou = IOU(self.anchors, train_boxes[prev_index:prev_index+train_num_boxes[i]])
            
            panchor_pos = torch.unique(torch.cat((torch.where(iou>=0.7)[0], torch.argmax(iou, dim=0))))
            nanchor_pos = torch.unique(torch.where(iou<=0.3)[0])

            positive_anchors = self.anchors[panchor_pos]
            negative_anchors = self.anchors[nanchor_pos]
            
            p_temp = torch.where(torch.all(positive_anchors>=0, dim=1) & torch.all(positive_anchors<=self.img_shape[0], dim=1))[0] ## assuming square image, adjust later
            ppos = p_temp[torch.tensor(np.random.choice(len(p_temp), size=np.minimum(int(self.num_proposals_per_image/2), len(p_temp)), replace=False))]
            
            n_temp = torch.where(torch.all(negative_anchors>=0, dim=1) & torch.all(negative_anchors<=self.img_shape[0], dim=1))[0]
            npos = n_temp[torch.tensor(np.random.choice(len(n_temp), size=np.maximum(int(self.num_proposals_per_image/2), self.num_proposals_per_image-len(p_temp)), replace=False))]
            
            anchor_pos = panchor_pos[ppos] 
            anchor_proposals = positive_anchors[ppos]
            anchor_pos = torch.cat((anchor_pos, nanchor_pos[npos]), dim=0)
            anchor_proposals = torch.cat((anchor_proposals, negative_anchors[npos]), dim=0)
            
            anchor_class = torch.cat((torch.full((len(ppos), 1), 1), torch.full((len(npos), 1), 0)), dim=0).view(-1)
            anchor_gt_pos = torch.argmax(iou[anchor_pos], dim=1)
            anchor_gt = train_boxes[prev_index+anchor_gt_pos, :]
            anchor_gt_class = train_classes[prev_index+anchor_gt_pos]

            rand_perm = torch.randperm(self.num_proposals_per_image)
            anchor_proposals = anchor_proposals[rand_perm, :]
            anchor_class = anchor_class[rand_perm]
            anchor_pos = anchor_pos[rand_perm]
            anchor_gt = anchor_gt[rand_perm]
            anchor_gt_class = anchor_gt_class[rand_perm]
            
            anchor_proposals_batch.append(anchor_proposals)
            anchor_class_batch.append(anchor_class)
            anchor_pos_batch.append(anchor_pos)
            anchor_gt_batch.append(anchor_gt)
            anchor_gt_class_batch.append(anchor_gt_class)

            prev_index += train_num_boxes[i]
            
        anchor_proposals_batch = torch.stack(anchor_proposals_batch)
        anchor_class_batch = torch.stack(anchor_class_batch)
        anchor_pos_batch = torch.stack(anchor_pos_batch)
        anchor_gt_batch = torch.stack(anchor_gt_batch)
        anchor_gt_class_batch = torch.stack(anchor_gt_class_batch)

        return anchor_proposals_batch, anchor_class_batch, anchor_pos_batch, anchor_gt_batch, anchor_gt_class_batch