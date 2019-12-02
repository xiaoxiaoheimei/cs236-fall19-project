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
from network import model_face_rotate
from data import attributeDataset
from util.logger import logger
from optimizer import optim_face_normalize

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
    with open('info/celeba-train-normal.txt', 'r') as f:
        train_normal_list = [tmp.rstrip() for tmp in f]
    with open('info/celeba-train-tilt.txt', 'r') as f:
        train_tilt_list = [tmp.rstrip() for tmp in f]
    with open('info/celeba-test-normal.txt', 'r') as f:
        test_normal_list = [tmp.rstrip() for tmp in f]
    with open('info/celeba-test-tilt.txt', 'r') as f:
        test_tilt_list = [tmp.rstrip() for tmp in f]
    train_dataset = attributeDataset.GrouppedAttrLabelDataset(normal_list=train_normal_list,
                                                         tilt_list=train_tilt_list,
                                                         attributes=attr,
                                                         scale=(224, 224),
                                                         crop_size=(160, 160),
                                                         img_dir_path=img_dir,
                                                         landmark_dir_path=landmark_dir,
                                                         csv_path='info/celeba-with-orientation.csv',
                                                         label_path='info/img_label.csv',
                                                         random_crop_bias=0)
    test_dataset = attributeDataset.GrouppedAttrLabelDataset(normal_list=test_normal_list,
                                                         tilt_list=test_tilt_list,
                                                         attributes=attr,
                                                         scale=(224, 224),
                                                         crop_size=(160, 160),
                                                         img_dir_path=img_dir,
                                                         landmark_dir_path=landmark_dir,
                                                         csv_path='info/celeba-with-orientation.csv',
                                                         label_path='info/img_label.csv',
                                                         random_crop_bias=0)

    print('define dataloader')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=True)

    print('define model & optim')
    model = define_model()
    optimizer = define_optim(model)

    print('start training')
    # modified from training_framework.py run()
    global_step = 0
    e0 = 0
    if recover_step_epoch:  # recover the global step and the epoch
        if os.path.exists(optimizer.opt.save_dir + '/step_epoch.yaml'):
            log('recover_step_epoch')
            with open(optimizer.opt.save_dir + '/step_epoch.yaml', 'r') as f:
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
    optim = optim_face_normalize.optimizer(model)
    return optim


def define_model():
    encoder = resnet50.Resnet50_ft()
    decoder = model_face_rotate.decoder_face_normalize()
    # discrim_real = discriminator(in_channels=3)
    # discrim_lm = discriminator(in_channels=4)
    discrim_real = model_face_rotate.discriminator(in_channels=3, wgan=False)
    # discrim_lm = model_face_rotate.discriminator(in_channels=4, wgan=False)
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)
    discrim_real = nn.DataParallel(discrim_real)
    # discrim_lm = nn.DataParallel(discrim_lm)
    return encoder, decoder, discrim_real


if __name__ == '__main__':
    run()
