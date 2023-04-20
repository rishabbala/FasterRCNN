import torch
from torch import nn, Tensor
import torchvision
import torchvision.transforms as tf
import torchvision.models as models
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
from torch.utils.tensorboard import SummaryWriter


from dataloader import VOCDataset, collate_fn
from models.resnet import MakeResnet
from models.FPN import BBPredNet, ROIPredNet
from utils.box_utils import GenerateAnchorBoxes, IOU, gen_coord2yxhw

writer = SummaryWriter(log_dir='./runs')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CELoss_weighted = nn.CrossEntropyLoss(reduction='sum', weight=torch.FloatTensor([1, 1]).to(device))
CELoss = nn.CrossEntropyLoss(reduction='sum')
SmoothL1Loss = nn.SmoothL1Loss(beta=1, reduction='none')
relu = torch.nn.ReLU()

train_dataset = VOCDataset(set_type='train')
test_dataset = VOCDataset(set_type='test')

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)

model = MakeResnet([2, 2, 2, 2]).to(device)

# resnet18 = models.resnet18(pretrained=True)
# model = torch.nn.Sequential(
#     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
#     nn.BatchNorm2d(num_features=64),
#     nn.ReLU(inplace=True),
#     *(list(resnet18.children())[4:-2])
#     ).to(device)


generate_anchors = GenerateAnchorBoxes(device=device)
FPN = BBPredNet(generate_anchors.box_types).to(device)
roi_net = ROIPredNet().to(device)

model.load_state_dict(torch.load('./weights/resnet18_2380.pth'))
FPN.load_state_dict(torch.load('./weights/FPN_2380.pth'))
roi_net.load_state_dict(torch.load('./weights/roi_2380.pth'))

optim = torch.optim.Adam(list(model.parameters())+list(FPN.parameters())+list(roi_net.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2500, 2700, 2900], gamma=0.1)

for epoch in range(2390, 3001):
    total_loss = 0
    total_ce_loss = 0
    total_l1_loss = 0
    total_roi_pred_loss = 0
    for _, (train_images, train_boxes, train_num_boxes, train_classes) in enumerate(train_dataloader):

        optim.zero_grad()

        train_images, train_boxes, train_num_boxes, train_classes = train_images.to(device), train_boxes.to(device), train_num_boxes.to(device), train_classes.to(device)

        out = model(train_images)
        
        anchor_proposals_batch, anchor_class_batch, anchor_pos_batch, anchor_gt_batch, anchor_gt_class_batch = generate_anchors.generate_anchors_from_fmap(out.shape[-2:], train_boxes, train_num_boxes, train_classes)

        anchor_proposals_batch, anchor_class_batch, anchor_pos_batch, anchor_gt_batch, anchor_gt_class_batch = anchor_proposals_batch.to(device), anchor_class_batch.to(device), anchor_pos_batch.to(device), anchor_gt_batch.to(device), anchor_gt_class_batch.to(device)

        anchor_gt_yxhw_batch = gen_coord2yxhw(anchor_gt_batch)
        anchor_proposals_yxhw_batch = gen_coord2yxhw(anchor_proposals_batch)

        fmap_class, fmap_box_yx, fmap_box_hw = FPN(out)

        height_stride = train_images.shape[2]/out.shape[2]
        width_stride = train_images.shape[3]/out.shape[3]

        anchor_proposals_center_batch_fmap = torch.stack([anchor_proposals_yxhw_batch[:, :, 0]/height_stride, anchor_proposals_yxhw_batch[:, :, 1]/width_stride], dim=2)
        anchor_proposals_type_fmap = anchor_pos_batch % generate_anchors.box_types

        bbox_proposal_coord = torch.cat((anchor_proposals_type_fmap[:, :, None], anchor_proposals_center_batch_fmap), dim=2).to(torch.int32)

        fmap_bbox_stats_yx = []
        fmap_bbox_stats_hw = []
        fmap_class_pred = []
        roi_boxes = []
        roi_classes = []
        prev_index = 0

        for i in range(train_num_boxes.shape[0]):
            fmap_bbox_stats_yx.append(fmap_box_yx[i, :, bbox_proposal_coord[i, :, 0], bbox_proposal_coord[i, :, 1], bbox_proposal_coord[i, :, 2]].T)
            fmap_bbox_stats_hw.append(fmap_box_hw[i, :, bbox_proposal_coord[i, :, 0], bbox_proposal_coord[i, :, 1], bbox_proposal_coord[i, :, 2]].T)
            fmap_class_pred.append(fmap_class[i, :, bbox_proposal_coord[i, :, 0], bbox_proposal_coord[i, :, 1], bbox_proposal_coord[i, :, 2]].T)

            roi_index = torch.where(anchor_class_batch[i]==1)[0]

            temp_box = torch.stack([
                fmap_bbox_stats_yx[-1][roi_index, 1] - fmap_bbox_stats_hw[-1][roi_index, 1]/2,
                fmap_bbox_stats_yx[-1][roi_index, 0] - fmap_bbox_stats_hw[-1][roi_index, 0]/2,
                fmap_bbox_stats_yx[-1][roi_index, 1] + fmap_bbox_stats_hw[-1][roi_index, 1]/2,
                fmap_bbox_stats_yx[-1][roi_index, 0] + fmap_bbox_stats_hw[-1][roi_index, 0]/2
            ], dim=1)
            temp_classes = anchor_gt_class_batch[i, roi_index]

            # roi_boxes.append(train_boxes[prev_index:prev_index+train_num_boxes[i], [1, 0, 3, 2]])
            roi_boxes.append(temp_box)
            roi_classes.append(temp_classes)

            prev_index += train_num_boxes[i]

        fmap_bbox_stats_yx = torch.stack(fmap_bbox_stats_yx)
        fmap_bbox_stats_hw = torch.stack(fmap_bbox_stats_hw)
        fmap_class_pred = torch.stack(fmap_class_pred)
        roi_classes = torch.cat(roi_classes, dim=0)

        roi_pool = torchvision.ops.roi_pool(input=out, boxes=roi_boxes, output_size=(16, 16), spatial_scale=1/8)
        roi_pool_features = roi_net(roi_pool)

        # anchor_class_batch = anchor_class_batch.detach()
        # anchor_proposals_yxhw_batch = anchor_proposals_yxhw_batch.detach()
        # anchor_gt_yxhw_batch = anchor_gt_yxhw_batch.detach()

        ce_loss = CELoss_weighted(fmap_class_pred.view(-1, 2), anchor_class_batch.view(-1))/(train_num_boxes.shape[0] * generate_anchors.num_proposals_per_image)
        roi_pred_loss = CELoss(roi_pool_features, roi_classes)/(20 * roi_classes.shape[0])

        t_yx = (fmap_bbox_stats_yx - anchor_proposals_yxhw_batch[:, :, :2])/anchor_proposals_yxhw_batch[:, :, 2:]
        t_hw = torch.log(fmap_bbox_stats_hw) - torch.log(anchor_proposals_yxhw_batch[:, :, 2:])

        t_star_yx = (anchor_gt_yxhw_batch[:, :, :2] - anchor_proposals_yxhw_batch[:, :, :2])/anchor_proposals_yxhw_batch[:, :, 2:]
        t_star_hw = torch.log(anchor_gt_yxhw_batch[:, :, 2:]) - torch.log(anchor_proposals_yxhw_batch[:, :, 2:])

        l1_loss = SmoothL1Loss(torch.cat((t_yx, t_hw), dim=2), torch.cat((t_star_yx, t_star_hw), dim=2))
        l1_loss = torch.sum(l1_loss * anchor_class_batch.unsqueeze(2))/(generate_anchors.box_types * out.shape[1] * out.shape[2])

        loss = 0.4*ce_loss + 30*l1_loss + 0.1*roi_pred_loss

        loss.backward()
        optim.step()
        scheduler.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_l1_loss += l1_loss.item()
        total_roi_pred_loss += roi_pred_loss.item()

    writer.add_scalar('Training Loss', total_loss, epoch)
    writer.add_scalar('BBox Cross Entropy Loss', 0.2*total_ce_loss, epoch)
    writer.add_scalar('L1 Loss', 30*total_l1_loss, epoch)
    writer.add_scalar('ROI Class Prediction Loss', 0.1*total_roi_pred_loss, epoch)

    if epoch %10 == 0:
        torch.save(model.state_dict(), f"./weights/resnet18_{epoch}.pth")
        torch.save(FPN.state_dict(), f"./weights/FPN_{epoch}.pth")
        torch.save(roi_net.state_dict(), f"./weights/roi_{epoch}.pth")

    print(f"Total Loss: {total_loss}, Epoch: {epoch}")