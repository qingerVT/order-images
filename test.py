#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader, Dataset
import json
from skimage import io
from PIL import Image
import argparse
import logging
import sys

# load resnet to generate feature
resnet18 = models.resnet18(pretrained=True)
modules=list(resnet18.children())[:-1]
resnet18=nn.Sequential(*modules)
for p in resnet18.parameters():
    p.requires_grad = False
resnet18.eval()
resnet18.cuda()

from  orderImage import (
    OrderImageDataset,
    orderImageNet
)


def test(model, loader):
    loss = 0.0
    acc = 0.0
    for i, sample in enumerate(loader):
        frames = sample['frames'].cuda()
        orders = sample['frame_order'].cuda()
        batch_size, n_frame = frames.size(0), frames.size(1)
        # generate image features
        feat = resnet18(frames.reshape(-1, 3, 224, 224))
        feat = feat.reshape(batch_size, n_frame, -1)
        score, preds = model(feat)
        loss += model.loss(score, orders)
        acc += model.accuracy(preds, orders)
    print('loss : %s' % (loss/len(loader)).cpu().detach().numpy())
    print('accuracy : %s' % (acc/len(loader)).cpu().detach().numpy())

if __name__ == "__main__":
      
    # get arguments from command line
    my_parser = argparse.ArgumentParser(description='List the content of a folder')
    my_parser.add_argument('--nkernel', type=int, default=200, help='the path to list')
    my_parser.add_argument('--hidden', type=int, default=512, help='the path to list')
    my_parser.add_argument('--dropout', action="store_true", help='the path to list')
    my_parser.add_argument('--path', type=str, default='checkpoint_best.pt', help='the path to list')
    args = my_parser.parse_args()
    print(args)

    # build dataloader
    test_dataset =  OrderImageDataset("../data/godTorch/train.json")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)
    
    # build model
    model = orderImageNet(args.hidden, args.nkernel)
    
    # load checkpoint
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    
    # testing 
    test(model, test_loader)

