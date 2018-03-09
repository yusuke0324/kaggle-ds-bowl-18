import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import os
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from multiprocessing import Pool, cpu_count
from keras import backend as K
import tensorflow as tf
import random

TRAIN_PATH = '/home/ubuntu/work/kaggle-ds-bowl-18/data/stage1_train/'
TEST_PATH = '/home/ubuntu/work/kaggle-ds-bowl-18/data/stage1_test/'

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

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

def negpos_reverse(img, thresh=80):
    '''
    nega-posi reverse if the img average color is less than 80
    '''
    avg = get_average_color_nparray(img)
    if avg[0] < thresh:
        img = cv2.bitwise_not(img)

    return img

def equalize_hist(img):
    # in oder to support color image, here, convert RGB -> YUV and only Y is equalized then YUV -> RGB
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
    img_output = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)

    return img_output



def get_average_color_nparray(nparray):
    '''
    return average color as a nparray
    This is for anparray (height, width, channel)
    '''
    average_color_per_row = np.average(nparray, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    average_color = np.uint8(average_color)

    # convert to uint8
    return average_color



def convert_rgb_to_hsd_ndarray(rgb_ndarray):
    '''
    return rescaled hsd ndarray, (n, h, w, 3) from RGB ndarray.
    ARG:
        ndarray: (n, h, w, 3) values should be 0 ~ 255
    '''
    rgb_list = [rgb for rgb in rgb_ndarray]
    p = Pool(3)
    ret = p.map(convert_rgb_to_hsd, rgb_list)
    hsd_ndarray = np.asarray(ret)
    return hsd_ndarray

def convert_rgb_to_hsd(img):
    '''
    convert RGB image to HSD color space image
    ARG:
        img: (h, w, 3) shape's RGB channel ndarray
    RETURN:
        (h, w, 3) shape's Density, cx and cy cannel nd array
    '''
    img = np.array(img, dtype=np.uint16)
    r_max = img[:, :, 0].max()
    g_max = img[:, :, 1].max()
    b_max = img[:, :, 2].max()

    density_r = -np.log((img[:, :, 0] + 1)/(r_max+2))
    density_g = -np.log((img[:, :, 1] + 1)/(g_max+2))
    density_b = -np.log((img[:, :, 2] + 1)/(b_max+2))

    average_density = (density_r + density_g + density_b)/3

    cx = (density_r / average_density) - 1

    cy = (density_g - density_b) / (np.sqrt(3) * average_density)

    return np.dstack((average_density, cx, cy))

def random_crop(img, mask_img, flip=True, shape=(128, 128), subset=None, maxsubset=12):
    '''
    return random crop image.
    '''
    # set default subset
    img_h = img.shape[0]
    img_w = img.shape[1]

    # if the image shape is smaller than shape, return none
    if img_h < shape[0]:
        return None
    elif img_w < shape[1]:
        return None

    range_h = img_h - shape[0]
    range_w = img_w - shape[1]

    # crop_images = np.empty()
    crop_images = []
    mask_images = []

    if subset is None:
        subset = int(img_h * img_w/(shape[0] * shape[1]))
        subset = min(subset, maxsubset)

    for i in range(0, subset):
        # left upper pint coord
        fx = int(random.uniform(0, range_w))
        fy = int(random.uniform(0, range_h))
        crop_image = img[fy:fy+shape[0], fx:fx+shape[1]]
        crop_mask_image = mask_img[fy:fy+shape[0], fx:fx+shape[1]]
        # flip
        if flip:
            flip_v = random.choice([-1, 0, 1, 9])
            if flip_v != 9:
                crop_image = cv2.flip(crop_image, flip_v)
                crop_mask_image = cv2.flip(crop_mask_image, flip_v)
                # crop_norm_image = normalize_image(crop_image)
        crop_images.append(crop_image)
        mask_images.append(np.squeeze(crop_mask_image))
        # np.append(crop_images, crop_norm_image, axis=0)

    return crop_images, mask_images

def tile_predict(img, model, patch_shape=(128, 128)):

    patch_h = patch_shape[0]
    img_h = img.shape[0]


    patch_w = patch_shape[1]
    img_w = img.shape[1]

    y_list = [(i*patch_h, (i+1)*patch_h) for i in range(img_h//patch_h)]
    x_list = [(i*patch_w, (i+1)*patch_w) for i in range(img_w//patch_w)]

    # normalize and resize to match for the model input
    img = np.array(img, dtype='float32')
    img /= 255
    img = np.resize(img, (1, img_h, img_w, 3))

    predict = np.zeros((img_h, img_w, 1))
    for y in y_list:
        for x in x_list:
            crop = img[:, y[0]:y[1], x[0]:x[1], :]
            predict_crop = model.predict(crop)
            predict_crop_binary = (predict_crop > 0.5).astype(np.uint8)
            predict[y[0]:y[1], x[0]:x[1], :] = predict_crop_binary

    # return predict
    # predict the rest of the image
    # predict right edge
    for y in y_list:
        crop = img[:, y[0]:y[1], img_w-patch_w:img_w, :]
        predict_crop = model.predict(crop)
        predict_crop_binary = (predict_crop > 0.5).astype(np.uint8)
        predict[y[0]:y[1], img_w-patch_w:img_w, :] = predict_crop_binary

    # predict bottom edge
    for x in x_list:
        crop = img[:, img_h-patch_h:img_h, x[0]:x[1], :]
        predict_crop = model.predict(crop)
        predict_crop_binary = (predict_crop > 0.5).astype(np.uint8)
        predict[img_h-patch_h:img_h, x[0]:x[1], :] = predict_crop_binary

    return predict

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def make_submission_file(X_pred, test_ids, filename='submission.csv'):
    pass
    # test_ids = next(os.walk('./data/stage1_test'))[1]

    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(X_pred[n]))
        rles.extend(rle)
        new_test_ids.extend([id_]*len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('./submission/'+filename, index=False)

    # submit command
    # $kg submit sub-128-np-eq_3_9.csv -u 'yuusuke' -p 'paul0324' -c 'data-science-bowl-2018'
    return sub






















