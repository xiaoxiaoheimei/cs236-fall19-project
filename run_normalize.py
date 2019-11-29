import os
import sys
import pdb
import pandas
import scipy.misc as sci
import numpy as np
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from network import resnet50
from data import attributeDataset

sys.setrecursionlimit(100000)

def run():
    attr = 'Mouth_Slightly_Open@Smiling,Male@No_Beard@Mustache@Goatee@Sideburns,Black_Hair@Blond_Hair@Brown_Hair@Gray_Hair,Bald@Receding_Hairline@Bangs,Young'
    data_dir = '../cerebA/img_align_celeba'
    batch_size = 32
    epoch = 10

    with open('info/celeba-train.txt', 'r') as f:
        train_list = [os.path.join(data_dir, tmp.rstrip()) for tmp in f]
    with open('info/celeba-test.txt', 'r') as f:
        test_list = [os.path.join(data_dir, tmp.rstrip()) for tmp in f]
    train_dataset = attributeDataset.GrouppedAttrDataset(image_list=train_list, attributes=attr,
                                                         csv_path='info/celeba-with-orientation.csv',
                                                         label_path='info/img_label.csv',
                                                         random_crop_bias=8)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # pdb.set_trace()
    for e in range(epoch):
        for iter, data in enumerate(train_dataloader):
            # img, _, label = data
            img, attr = data
            print(img.shape, label.shape)


def define_model():
    encoder = resnet50_ft(weights_path='checkpoints/resnet50_face/resnet50_ft_dims_2048.pth')
    decoder = decoder_normalize()
    return encoder, decoder


class decoder_normalize(nn.Module):
    def __init__(self):
        super(decoder_normalize, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(2048, 512, kernel_size=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=256),
            deconv_block(in_channels=256),
            res_block(in_channels=256, out_channels=128),
            deconv_block(in_channels=128),
            res_block(in_channels=128, out_channels=64),
            deconv_block(in_channels=64),
            res_block(in_channels=64, out_channels=32),
            deconv_block(in_channels=32),
            res_block(in_channels=32, out_channels=32),
            deconv_block(in_channels=32),
            nn.Conv2d(32, 3, kernel_size=1, padding=1),
            nn.Sigmoid(),
        ])
    def forward(self, x):
        y = self.model(x)
        return y


class res_block(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(res_block, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ])
    def forward(self, x):
        y = self.model(x)
        return y

class deconv_block(nn.Module):
    def __init__(self, in_channels=256, upsample_mode='nearest'):
        super(res_block, self).__init__()
        self.model = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        ])
    def forward(self, x):
        y = self.model(x)
        return y

if __name__ == '__main__':
    run()
