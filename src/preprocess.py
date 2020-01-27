import shutil
import os
import pathlib
import zipfile
from glob import glob
import logging
import warnings

import numpy as np 
import pandas as pd
import imutils
import cv2

import tensorflow as tf

HEIGHT = 64
WIDTH = 1024 

# PREPROCESS IMAGES BEFORE TRAINING
def color_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5,5), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    return image

def resize_padding(image):
    h, w = image.shape
    if h > HEIGHT:
      image = imutils.resize(image, height = HEIGHT)

    h, w = image.shape
    if w > WIDTH:
      image = imutils.resize(image, width = WIDTH)

    h, w = image.shape
    if h < HEIGHT: 
      add_zeros = np.zeros((HEIGHT-h, WIDTH))
      image = np.concatenate((image, add_zeros), axis=0)

    h, w = image.shape  
    if w < WIDTH:
      add_zeros = np.zeros((HEIGHT, WIDTH-w))
      image = np.concatenate((image, add_zeros), axis=1)

    return image

def remove_cursive_style(img):
    """Remove cursive writing style from image with deslanting algorithm
    
    Deslating image process based in,
    A. Vinciarelli and J. Luettin,
    A New Normalization Technique for Cursive Handwritten Wrods, in
    Pattern Recognition, 22, 2001.
    """

    def calc_y_alpha(vec):
        indices = np.where(vec > 0)[0]
        h_alpha = len(indices)

        if h_alpha > 0:
            delta_y_alpha = indices[h_alpha - 1] - indices[0] + 1

            if h_alpha == delta_y_alpha:
                return h_alpha * h_alpha
        return 0

    alpha_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    rows, cols = img.shape
    results = []

    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = otsu if ret < 127 else sauvola(img, (int(img.shape[0] / 2), int(img.shape[0] / 2)), 127, 1e-2)

    for alpha in alpha_vals:
        shift_x = max(-alpha * rows, 0.)
        size = (cols + int(np.ceil(abs(alpha * rows))), rows)
        transform = np.asarray([[1, alpha, shift_x], [0, 1, 0]], dtype=np.float)

        shear_img = cv2.warpAffine(binary, transform, size, cv2.INTER_NEAREST)
        sum_alpha = 0
        sum_alpha += np.apply_along_axis(calc_y_alpha, 0, shear_img)
        results.append([np.sum(sum_alpha), size, transform])

    result = sorted(results, key=lambda x: x[0], reverse=True)[0]
    warp = cv2.warpAffine(img, result[2], result[1], borderValue=255)

    return cv2.resize(warp, dsize=(cols, rows))

def preprocess_raw(path, input_size, save_type=False):
    """ Preprocess images before passing into tf.data.Dataset
        Images are processed by OpenCV then convert to tensor datatype.
        Input: path to image file
        Output: image in tensor datatype with shape (64, 1024, 1)
    """
    # Read and preprocess
    image = cv2.imread(path)
    image = color_preprocess(image)
    image = remove_cursive_style(image)
    image = resize_padding(image)
    
    # Save image to folder
    if isinstance(save_type, str):
        file_name = pathlib.Path(path).name
        cv2.imwrite(os.path.join('..', 'data', save_type, file_name), image)
    
    return image

# OPTIONAL AUGMENTATION
def cv2_augmentation(imgs,
                 rotation_range=0,
                 scale_range=0,
                 height_shift_range=0,
                 width_shift_range=0,
                 dilate_range=1,
                 erode_range=1):
    """Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)"""

    imgs = imgs.astype(np.float32)
    _, h, w = imgs.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    for i in range(len(imgs)):
        imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
        imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

    return imgs

@tf.function
def augmentation(image, label):
    image = tf.py_function(cv2_augmentation, [image], tf.float32)
    return image, label


