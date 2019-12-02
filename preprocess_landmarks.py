#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import scipy as sci
from skimage import draw


# In[105]:


# data_path = '/Volumes/Padlock/cerebA/'
data_path = '/data/jiahong/CS236/cerebA/'
landmark_txt_path = data_path + 'Anno/list_landmarks_align_celeba.txt'

with open(landmark_txt_path) as f:
    landmarks_data = f.readlines()
landmarks_data = [x.strip() for x in landmarks_data][2:]

landmarks = np.zeros((len(landmarks_data), 10)).astype(int)
file_list = []
for i, x in enumerate(landmarks_data):
    file_list.append(x.split()[0])
    x = x.split()[1:]
    for j in range(10):
        landmarks[i,j] = x[j]


# In[65]:


# compute mean landmarks
landmarks_mean = landmarks.mean(axis=0).astype(int)
print('Mean: ', landmarks_mean)
landmarks_var = landmarks.var(axis=0).astype(int)
print('Var: ', landmarks_var)


# In[70]:


# classify frontal or not by nose landmarks
thres = 30
landmarks_diff = landmarks - landmarks_mean
frontal_label = np.zeros((landmarks.shape[0],))
for i in range(landmarks.shape[0]):
    if landmarks_diff[i, 4] ** 2 + landmarks_diff[i, 5] ** 2 > thres:
        frontal_label[i] = 1


# In[75]:


# make dict and save to csv
label_dict = {}
assert(len(file_list) == frontal_label.shape[0])
assert(len(file_list) == landmarks.shape[0])
label_dict['file_name'] = file_list
# label_dict['landmarks'] = landmarks
label_dict['label'] = frontal_label
df = pd.DataFrame(label_dict)
df.to_csv(data_path + 'Anno/img_label.csv', index=False)


# In[71]:


# visualization of landmarks
img_ori_path = data_path + 'img_align_celeba/'
img_dst_path = data_path + 'img_align_celeba_anno/'
for i in range(100):
    img_num = format(i+1, '06d')
    img_path = img_ori_path + img_num + '.jpg'
    img = sci.misc.imread(img_path)
    rr_all = np.empty(0,).astype(int)
    cc_all = np.empty(0,).astype(int)
    rr_mean = np.empty(0,).astype(int)
    cc_mean = np.empty(0,).astype(int)
    for j in range(5):
        rr, cc = draw.circle(landmarks[i,2*j+1], landmarks[i,2*j], radius=2, shape=img.shape)
        rr_all = np.concatenate([rr_all, rr])
        cc_all = np.concatenate([cc_all, cc])
        rr, cc = draw.circle(landmarks_mean[2*j+1], landmarks_mean[2*j], radius=2, shape=img.shape)
        rr_mean = np.concatenate([rr_mean, rr])
        cc_mean = np.concatenate([cc_mean, cc])
    img[rr_all, cc_all] = [255, 0, 0]
    img[rr_mean, cc_mean] = [0, 255, 0]
    if frontal_label[i] == 1:
        rr, cc = draw.circle(5, 5, radius=5, shape=img.shape)
        img[rr, cc] = [0, 255, 0]
    img_path = img_dst_path + img_num + '.jpg'
    sci.misc.imsave(img_path, img)


# In[126]:


# save landmark img for all images
img_dst_path = data_path + 'img_landmark/'
for i, marks in enumerate(landmarks):
    img_num = format(i+1, '06d')
    img = np.zeros((218, 178))
    rr_all = np.empty(0,).astype(int)
    cc_all = np.empty(0,).astype(int)
    for j in range(5):
        rr, cc = draw.circle(landmarks[i,2*j+1], landmarks[i,2*j], radius=5, shape=img.shape)
        rr_all = np.concatenate([rr_all, rr])
        cc_all = np.concatenate([cc_all, cc])
    img[rr_all, cc_all] = 1
    img_path = img_dst_path + img_num + '.jpg'
    sci.misc.imsave(img_path, img)
    if i % 1000 == 0:
        print('save landmark images, ', i)


# In[131]:


# save mean landmarks
img = np.zeros((218, 178))
rr_mean = np.empty(0,).astype(int)
cc_mean = np.empty(0,).astype(int)
for j in range(5):
    rr, cc = draw.circle(landmarks_mean[2*j+1], landmarks_mean[2*j], radius=5, shape=img.shape)
    rr_mean = np.concatenate([rr_mean, rr])
    cc_mean = np.concatenate([cc_mean, cc])
img[rr_mean, cc_mean] = 1
img_path = img_dst_path + 'mean.jpg'
sci.misc.imsave(img_path, img)
