import os
import sys
import pdb
import pandas
import scipy.misc as sci
import numpy as np
# import tqdm
from tqdm import tqdm
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from network import resnet50
from data import attributeDataset
from util.logger import logger
from optimizer import optim_face_rotate

# sys.setrecursionlimit(100000)
log = logger(True)

def run():
    attr = 'Mouth_Slightly_Open@Smiling,Male@No_Beard@Mustache@Goatee@Sideburns,Black_Hair@Blond_Hair@Brown_Hair@Gray_Hair,Bald@Receding_Hairline@Bangs,Young'
    img_dir = '../cerebA/img_align_celeba'
    landmark_dir = '../cerebA/img_landmark'
    batch_size = 64
    epoch = 100
    recover_step_epoch = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

    print('define dataset')
    with open('info/celeba-train.txt', 'r') as f:
        train_list = [tmp.rstrip() for tmp in f]
    with open('info/celeba-test.txt', 'r') as f:
        test_list = [tmp.rstrip() for tmp in f]
    train_dataset = attributeDataset.GrouppedAttrLandmarkDataset(image_list=train_list, attributes=attr,
                                                         scale=(224, 224),
                                                         crop_size=(160, 160),
                                                         img_dir_path=img_dir,
                                                         landmark_dir_path=landmark_dir,
                                                         csv_path='info/celeba-with-orientation.csv',
                                                         label_path='info/img_label.csv',
                                                         random_crop_bias=0)
    test_dataset = attributeDataset.GrouppedAttrLandmarkDataset(image_list=test_list, attributes=attr,
                                                         scale=(224, 224),
                                                         crop_size=(160, 160),
                                                         img_dir_path=img_dir,
                                                         landmark_dir_path=landmark_dir,
                                                         csv_path='info/celeba-with-orientation.csv',
                                                         label_path='info/img_label.csv',
                                                         random_crop_bias=0)

    print('define dataloader')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0, drop_last=True)

    print('define model & optim')
    model = define_model()
    optimizer = define_optim(model)

    print('start training')
    # modified from training_framework.py run()
    global_step = 0
    e0 = 0
    if recover_step_epoch:  # recover the global step and the epoch
        if os.path.exists(optim.opt.save_dir + '/step_epoch.yaml'):
            log('recover_step_epoch')
            with open(optim.opt.save_dir + '/step_epoch.yaml', 'r') as f:
                global_step_epoch = yaml.load(f)
                global_step = int(global_step_epoch['global_step'])
                e0 = int(global_step_epoch['epoch'])
    for e in range(e0, epoch):
        print(e)
        # for iter, data in enumerate(train_dataloader):
        for iter, data in enumerate(tqdm(train_dataloader)):
            optimizer.set_input(data)
            optimizer.optimize_parameters(global_step)
            if global_step % 10 == 0:
                tqdm.write(optimizer.print_current_errors(e, iter, record_file=optimizer.opt.save_dir,
                                                               print_msg=False))
                optimizer.add_summary(global_step)
            if (global_step) % 100 == 0:
                log('save samples ', global_step)
                optimizer.save_samples(test_dataloader, global_step)
            if global_step > 0 and global_step % 2000 == 0:
                optimizer.save()
            global_step += 1
            with open(optimizer.opt.save_dir + '/step_epoch.yaml', 'w') as f:
                global_step_epoch = {}
                global_step_epoch['global_step'] = global_step
                global_step_epoch['epoch'] = e
                yaml.dump(global_step_epoch, f, default_flow_style=False)
        #     if iter == 10:
        #         break
        # if e == 2:
        #     break
        if e % 10 == 0 and e > 0:
            optimizer.save(e)
        optimizer.add_summary_heavy(e)
    optimizer.save()


def define_optim(model):
    optim = optim_face_rotate.optimizer(model)
    return optim


def define_model():
    encoder = resnet50.Resnet50_ft()
    decoder = decoder_face_rotate()
    # discrim_real = discriminator(in_channels=3)
    # discrim_lm = discriminator(in_channels=4)
    discrim_real = discriminator(in_channels=3, wgan=False)
    discrim_lm = discriminator(in_channels=4, wgan=False)
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)
    discrim_real = nn.DataParallel(discrim_real)
    discrim_lm = nn.DataParallel(discrim_lm)
    return encoder, decoder, discrim_real, discrim_lm

class discriminator(nn.Module):
    def __init__(self, in_channels=3, wgan=True):
        super(discriminator, self).__init__()
        self.wgan = wgan
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)])
        self.fn = nn.Linear(256*6*6, 1)

    def forward(self, x):
        # print('input', x.shape)
        bs = x.shape[0]
        x = self.model(x)
        if self.wgan == True:
            x = self.fn(x.view(bs,-1))
        return x


class decoder_face_rotate(nn.Module):
    def __init__(self):
        super(decoder_face_rotate, self).__init__()
        self.lm_encoder = nn.Sequential(*[
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()])

        self.model = nn.Sequential(*[
            nn.Conv2d(2048+256, 512, kernel_size=1, padding=0),
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
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            # nn.Tanh(),
        ])

    def forward(self, feat_id, lm):
        feat_lm = self.lm_encoder(lm)
        feat = torch.cat([feat_id, feat_lm], dim=1)
        y = self.model(feat)
        # return (y + 1) * 0.5
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
        super(deconv_block, self).__init__()
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
