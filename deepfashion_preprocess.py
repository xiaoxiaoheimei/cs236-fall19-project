import numpy as np
import pandas as pd
import pdb

# data_path = '/Volumes/Padlock/Category and Attribute Prediction Benchmark/'
data_path = '/data/jiahong/CS236/Category and Attribute Prediction Benchmark/'
# data_path = '../Category and Attribute Prediction Benchmark/'

# make info/deepfashion.csv by list_attr_img.txt and attribute_name.txt
if True:
    attr_img_txt_path = data_path + 'Anno/list_attr_img.txt'
    attr_cloth_txt_path = data_path + 'Anno/list_attr_cloth.txt'

    # read list_attr_cloth
    with open(attr_cloth_txt_path) as f:
        attr_cloth = f.readlines()
    attr_cloth = [x.strip() for x in attr_cloth][2:]
    attr_cloth_name = ['_'.join(x.split()[:-1]) for x in attr_cloth]
    attr_cloth_type = [int(x.split()[-1]) for x in attr_cloth]

    # save attribute_name.txt
    with open(data_path + 'info/deepfashion_attribute_name.txt', 'w') as f:
        for i, attr_name in enumerate(attr_cloth_name):
            f.write(attr_name)
            if i < 1000-1:
                f.write(',')

    # save grouped attribute
    attr_cloth_sorted = [[x,y] for y,x in sorted(zip(attr_cloth_type,attr_cloth_name))]
    attribute_groups = [[],[],[],[],[]]
    for attr in attr_cloth_sorted:
        attribute_groups[attr[1]-1].append(attr[0])
    attribute_groups = ['@'.join(attr_group) for attr_group in attribute_groups]

    with open(data_path + 'info/deepfashion_attribute_grouped.txt', 'w') as f:
        for i, attr_group in enumerate(attribute_groups):
            f.write(attr_group)
            if i < 5-1:
                f.write(',')

    # save deepfashion.csv
    with open(attr_img_txt_path) as f:
        attr_img = f.readlines()
    attr_img = [x.strip() for x in attr_img][2:]
    attr_img = [x.split() for x in attr_img]

    attr_cloth_name.insert(0,'name')
    attr_img_df = pd.DataFrame(attr_img, columns=attr_cloth_name)
    orientation = ['left' for i in range(len(attr_img))]
    attr_img_df['orientation'] = orientation
    print(attr_img_df)
    attr_img_df.to_csv(data_path + 'info/deepfashion.csv')

# split training and testing dataset
if True:
    # read list_attr_cloth
    eval_partition_txt_path = data_path + 'Eval/list_eval_partition.txt'
    with open(eval_partition_txt_path) as f:
        eval_partition = f.readlines()
    eval_partition = [x.strip() for x in eval_partition][2:]
    # eval_train = [x.split()[0] for x in eval_partition if x.split()[1] == 'train']
    # eval_test = [x.split()[0] for x in eval_partition if x.split()[1] == 'test']
    with open(data_path + 'info/deepfashion-train.txt', 'w') as f_train, open(data_path + 'info/deepfashion-test.txt', 'w') as f_test:
        for x in eval_partition:
            if x.split()[1] == 'train':
                f_train.write(x.split()[0]+'\n')
            elif x.split()[1] == 'test':
                f_test.write(x.split()[0]+'\n')
