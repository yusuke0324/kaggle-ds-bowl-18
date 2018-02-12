import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import os
from tqdm import tqdm


TRAIN_PATH = '/home/ubuntu/work/kaggle-ds-bowl-18/data/stage1_train/'
TEST_PATH = '/home/ubuntu/work/kaggle-ds-bowl-18/data/stage1_test/'

def rgbread(url):
    img = cv2.imread(url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_train(train_path='/home/ubuntu/work/kaggle-ds-bowl-18/data/stage1_train/'):
    
    train_ids = next(os.walk(train_path))[1]
    X_train = []
    Y_train = []
    
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = train_path + id_
        img = rgbread(path + '/images/' + id_ + '.png')
        X_train.append(img)
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (img.shape[0], img.shape[1]), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train.append(mask)
    
    return X_train, Y_train

def get_test(test_path='/home/ubuntu/work/kaggle-ds-bowl-18/data/stage1_test/'):
    test_ids = next(os.walk(test_path))[1]
    X_test = []
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = test_path + id_
        img = rgbread(path + '/images/' + id_ + '.png')
        X_test.append(img)
    return X_test

def multi_maskshow(X_train, Y_train, max_counter=150):
    counter = 0
    n = 4
    for i, img in enumerate(X_train):
        if counter % n == 0:
            plt.figure(figsize=(14, 6))
            
        plt.subplot(1, n, counter % n + 1)
        plt.imshow(img)
        plt.subplot(1, n , counter % n + 2)
        plt.imshow(np.squeeze(Y_train[i]))
        counter += 2
        
        if counter > max_counter:
            break

def multi_imshow(img_list, max_counter=100):
    counter = 0
    n = 4
    for i, img in enumerate(img_list):
        if counter % n == 0:
            plt.figure(figsize=(14, 6))
        
        plt.subplot(1, n, counter % n + 1)
        plt.imshow(img)
        counter += 1
        
        if counter > max_counter:
            break
                       