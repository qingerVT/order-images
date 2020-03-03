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

class OrderImageDataset(Dataset):
    """dataset called by dataloader
       sample contrains two keys: 1) frames: 5 images and 2) frame_order, e.g., [0, 3, 4, 2, 1]
    """
    def __init__(self, json_file):
        self.videos = []
        transform = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        data = json.loads(open(json_file).read())
        for k, v in data.items():
            ims = torch.zeros(5, 3, 224, 224)
            for j in range(len(v["frames"])):
                im = Image.open(os.path.join('../data/godTorch/images/', v["frames"][j]))
                ims[j] = transform(im)
            sample = {"frames": ims, "frame_order": torch.tensor(v["frame_order"])}
            self.videos.append(sample)

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.videos[idx]  

class CNN(nn.Module):
    """Conv network to extract contextual representation over a set of (sentences, images, etc)
       please check "Yoon Kim. Convolutional neural networks for sentence classification.CoRR, abs/1408.5882, 2014.URLhttp://arxiv.org/abs/1408.5882"
    """
    def __init__(self, in_dim, n_kernel, Ks, out_dim=None):
        super().__init__()
        if not out_dim:
            out_dim = in_dim
        self.convs1 = nn.ModuleList([nn.Conv2d(1, n_kernel, (K, in_dim)) for K in Ks])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(Ks) * n_kernel, int(len(Ks) * n_kernel))
        self.fc2 = nn.Linear(int(len(Ks) * n_kernel), out_dim)
        self.max_K = max(Ks)
    
    def forward(self, x):
        """
        input x: batch_size * 5 * feature-dim (e.g, 512)
        """
        x = x.unsqueeze(1)
        N, Ci, n_image, in_dim = x.size()
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return self.fc2(F.relu(self.fc1(x)))
    
def score2order(score):
    """Convert score to order, i.e., sorting
       For example, score [0.2, 3, -0.1] -> [1, 0, 2]
    """
    order = torch.zeros(score.size())
    _, index = score.sort(dim=-1, descending=True)
    for i in range(score.size(-1)):
        order[:, i] = (index == i).nonzero()[:, 1]
    return order

def extract_index(orders):
    """
    Convert order to a list of indexs with order [0, 1, 2, ...], where ids[i]=j and orders[j]=i
    For example, order [2, 1, 0] -> [2, 1, 0]
    """
    ids = []
    n = orders.size(-1)
    for i in range(n):
        id = torch.nonzero(orders == i)
        ids.append(id)
    return ids

def hinge_loss(preds, low, high):
    """Hinge loss (typically used in SVMs)
       Enforce score to be at least \delta far away. For example, O(I_i) < O(I_j) -> s(I_i) - s(I_j) > \delta
       In this case, we simply define \delta=1
    """
    return F.relu(1+preds[high[:,0], high[:,1]] - preds[low[:,0], low[:,1]]).mean()
     
        
class orderImageNet(nn.Module):
    """Produce score for an image given a set of images"""
    def __init__(self, hidden_dim=512, n_kernel=200, dropout=False):
        super().__init__()
        self.cnn = CNN(512, n_kernel, [2, 3, 5])
        self.fc0 = nn.Linear(2*512, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
   
    def forward(self, feats):
        # generate contextual representation
        CI = self.cnn(feats)
        # concatenate image feature and Contextual representation
        CI = CI.unsqueeze(1).expand_as(feats)
        feats = torch.cat((feats, CI), 2)
        if self.dropout: 
            feats = nn.Dropout(0.1)(feats)
        # go though MLPs
        feats = F.relu(self.fc0(feats))
        score = self.fc1(feats).squeeze(-1)
        # convert score to order prediction
        preds = score2order(score)
        return score, preds.long().cuda()
    
    def loss(self, preds, orders):
        """ Aggregate hinge loss with 4 pairs: (0, 1), (1, 2), (2, 3) and (3, 4)
            Ideally, there are 5^2 ordered pairs. We simply use 4
        """
        pairs = extract_index(orders)
        loss = 0.0
        for i in range(len(pairs)-1):
            low = pairs[i]
            high = pairs[i+1]
            loss += hinge_loss(preds, low, high)
        return loss/(len(pairs)-1)
    
    def accuracy(self, preds, orders):
        """ Calculate the percetage of images being correclty ordered
        """
        fp = (preds == orders).sum()
        return fp.float()/orders.numel()        

# unittest
class unittest(object):
    def __init__(self):
        super().__init__()
        
    def test_score2order(self):
        score = torch.tensor([[0.2, 0.1, 2], 
                              [-2, 0, -1],
                              [2, -1, 0]])
        real_order = torch.tensor([[1, 2, 0], 
                              [2, 0, 1],
                              [0, 2, 1]]).float()
        preds = score2order(score)
        assert torch.equal(preds, real_order), "score2order function failed"
        print("TEST PASSED: score2order function")

    def test_extract_index(self):
        orders = torch.tensor([[0, 2, 1], 
                              [2, 0, 1],
                              [1, 2, 0]])
        real_pairs = [torch.tensor([[0, 0], 
                                    [1, 1],
                                    [2, 2]]),
                     torch.tensor([[0, 2], 
                                   [1, 2],
                                   [2, 0]]),
                     torch.tensor([[0, 1], 
                                   [1, 0],
                                   [2, 1]])]
        preds = extract_index(orders)
        assert torch.equal(preds[0], real_pairs[0])
        assert torch.equal(preds[1], real_pairs[1])
        assert torch.equal(preds[2], real_pairs[2])
        print("TEST PASSED: extract_index function")
        
    def test_hinge_loss(self):
        score = torch.tensor([[0.3, 0.5, -0.2], 
                              [2.5, -0.2, 0.5],
                              [1.2, -2.0, 0.2]])
        real_pairs = [torch.tensor([[0, 0], 
                                    [1, 1],
                                    [2, 2]]),
                     torch.tensor([[0, 2], 
                                   [1, 2],
                                   [2, 0]]),
                     torch.tensor([[0, 1], 
                                   [1, 0],
                                   [2, 1]])]
        real_element = torch.tensor([[-0.2-0.3], 
                                     [0.5+0.2],
                                     [1.2-0.2]])
        real_loss = torch.tensor((0.5 + 1.7 + 2.0)/3)
        loss = hinge_loss(score, real_pairs[0], real_pairs[1])
        assert torch.isclose(loss, real_loss), "real_loss: %s, loss: %s" % (real_loss, loss)
        #
        real_element = torch.tensor([[0.5+0.2], 
                                     [2.5-0.5],
                                     [-2.0-1.2]])
        real_loss = torch.tensor((1.7 + 3.0 + 0)/3)
        loss = hinge_loss(score, real_pairs[1], real_pairs[2])
        assert torch.isclose(loss, real_loss), "real_loss: %s, loss: %s" % (real_loss, loss)
        print("TEST PASSED: hinge loss function")
    
    def test_model_loss(self):
        orders = torch.tensor([[0, 2, 1], 
                      [2, 0, 1],
                      [1, 2, 0]])
        
        score = torch.tensor([[0.3, 0.5, -0.2], 
                      [2.5, -0.2, 0.5],
                      [1.2, -2.0, 0.2]])
        model = orderImageNet()
        real_loss = torch.tensor((4.2+4.7)/3/2)
        loss = model.loss(score, orders)
        assert torch.isclose(loss, real_loss), "real_loss: %s, loss: %s" % (real_loss, loss)
        print("TEST PASSED: model loss")
    
    def test_model_accuracy(self):
        orders = torch.tensor([[0, 2, 1], 
                      [2, 0, 1],
                      [1, 2, 0]])
        preds =  torch.tensor([[1, 0, 2], 
                      [0, 2, 1],
                      [0, 2, 1]])
        real_acc = 2.0/9
        model = orderImageNet()
        acc = model.accuracy(preds, orders)
        assert acc == real_acc
        print("TEST PASSED: model accuracy function")

    
    def main(self):
        self.test_score2order()
        self.test_extract_index()
        self.test_hinge_loss()
        self.test_model_loss()
        self.test_model_accuracy()

test = unittest()
test.main()

def train(model, train_loader, val_loader, optimizer, scheduler, shuffle=False, folder=None, max_epoch=20):
    best_val_loss = 2000
    shuffle_prob = 0.9
    for epoch in range(max_epoch):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
                train_loss = 0.0
            else:
                model.eval()
                loader = val_loader
                val_loss = 0.0
                
            for i, sample in enumerate(loader):
                if phase == 'train':
                    optimizer.zero_grad()
                frames = sample['frames'].cuda()
                orders = sample['frame_order'].cuda()
                # shuffle images in a set to mitigate overfitting
                if phase == 'train' and np.random.rand() > shuffle_prob and shuffle:
                    aa = np.array([0, 1, 2, 3, 4])
                    np.random.shuffle(aa)
                    frames = frames[:,aa]
                    orders = orders[:,aa]
                batch_size, n_frame = frames.size(0), frames.size(1)
                # generate image features
                feat = resnet18(frames.reshape(-1, 3, 224, 224))
                feat = feat.reshape(batch_size, n_frame, -1)
                # forward
                score, preds = model(feat)
                if phase == 'train':
                    loss = model.loss(score, orders)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss
                else:
                    # acc = model.accuracy(preds, orders)
                    loss = model.loss(score, orders)
                    val_loss += loss
            if phase == 'train':
                logging.info('[Train] loss at epoch %s: %s' % (epoch, (train_loss/len(loader)).cpu().detach().numpy()))
            else:
                logging.info('[Validation] loss at epoch %s: %s' % (epoch, (val_loss/len(loader)).cpu().detach().numpy()))
        
        # save checkpoint
        if epoch % 20 == 0:
            ## increasing the probability of shuffling images over every 20 epochs
            shuffle_prob *= 0.9 
            torch.save({'state_dict': model.state_dict()}, '%s/checkpoint%s.pt' % (folder, epoch))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save to checkpoint_best.pt
            torch.save({'state_dict': model.state_dict()}, '%s/checkpoint_best.pt' % folder)

if __name__ == "__main__":
      
    # get arguments from command line
    my_parser = argparse.ArgumentParser(description='List the content of a folder')
    my_parser.add_argument('--lr', type=float, default=0.001, help='the path to list')
    my_parser.add_argument('--bsz', type=int, default=100, help='the path to list')
    my_parser.add_argument('--nkernel', type=int, default=200, help='the path to list')
    my_parser.add_argument('--maxepoch', type=int, default=200, help='the path to list')
    my_parser.add_argument('--shuffle', action='store_true', help='the path to list')
    my_parser.add_argument('--dropout', action='store_true', help='the path to list')
    my_parser.add_argument('--hidden', type=int, default=128, help='the path to list')
    args = my_parser.parse_args()
    print(args)

    # logging
    folder = "".join(sys.argv[1:]).replace('--', '')
    os.mkdir(folder)
    logging.basicConfig(filename=folder+'/train.log', level=logging.INFO)
    
    # build dataloader
    train_dataset =  OrderImageDataset("../data/godTorch/train.json")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bsz, shuffle=True)
    test_dataset =  OrderImageDataset("../data/godTorch/test.json")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bsz, shuffle=True)
    
    # build model
    model = orderImageNet(args.hidden, args.nkernel, args.dropout)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    
    # training 
    train(model, train_loader, test_loader, optimizer, scheduler, shuffle=args.shuffle, folder=folder, max_epoch=args.maxepoch)


# In[ ]:




