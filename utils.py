import os
import tensorflow as tf
import numpy as np
import math
import cv2
import random
import pathlib
import PIL.Image as pil_image

def load_image(path):
    return pil_image.open(path).convert('RGB')


def generate_lr(image, scale):
    w = image.width
    h = image.height
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    image = image.resize((w,h), resample=pil_image.BICUBIC)
    return image


def modcrop(image, modulo, train=True):
    
    if train:
        w = image.width - image.width % modulo
        h = image.height - image.height % modulo
        return image.crop((0, 0, w, h))
    else:
        w = (image.width//16)*16 
        h = (image.height//16)*16
        return image.crop((0, 0, w, h))


def generate_patch(image, patch_size, stride):
    for i in range(0, image.height - patch_size + 1, stride):
        for j in range(0, image.width - patch_size + 1, stride):
            yield image.crop((j, i, j + patch_size, i + patch_size))


def image_to_array(image):
    return np.array(image).transpose((2, 0, 1))


def normalize(x):
    return x / 255.0

def rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def PSNR(a, b, max=255.0, shave_border=0):
    a = a[shave_border:a.shape[0]-shave_border, shave_border:a.shape[1]-shave_border]
    b = b[shave_border:b.shape[0]-shave_border, shave_border:b.shape[1]-shave_border]
    return 10. * np.log10((max ** 2) / np.mean(((a - b) ** 2)))

def SSIM(a, b, max=255.0, shave_border=0):
    a = a[shave_border:a.shape[0]-shave_border, shave_border:a.shape[1]-shave_border]
    b = b[shave_border:b.shape[0]-shave_border, shave_border:b.shape[1]-shave_border]
    return tf.image.ssim(tf.expand_dims(a,axis=-1), tf.expand_dims(b,axis=-1), max_val=1.)

def concat_ycrcb(y, crcb):
    return np.concatenate((y, crcb), axis=2)

def convert_rgb_to_ycbcr(img):
    y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
    cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
    cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr])

def convert_ycbcr_to_rgb(img):
    r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921/256.
    g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576/256.
    b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836/256.
    return np.array([r, g, b])

def ics(img1, img2):
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma1 = np.sqrt(np.abs(sigma1_sq))
    sigma2 = np.sqrt(np.abs(sigma2_sq))
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    i = (2*mu1*mu2+C1)/(mu1_sq+mu2_sq+C1)
    c = (2*sigma1*sigma2+C2)/(sigma1_sq+sigma2_sq+C2)
    s = (sigma12+C2/2)/(sigma1*sigma2+C2/2)
    return i*c*s

def preprocess(path, scale):
    hr_patches = []
    lr_patches = []

    patch_size = 128
    stride = 84
        
    for i, image_path in enumerate(path):

        hr = load_image(image_path)
        hr = modcrop(hr, scale, train=True)
        lr = generate_lr(hr, scale)

        for patch in generate_patch(hr, patch_size, stride):
            rgb_mean = np.mean(patch)
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            patch = np.transpose(patch, (1,2,0))
            patch = patch-(rgb_mean/255)
            hr_patches.append(patch)

        for patch in generate_patch(lr, patch_size, stride):
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.float32), 'chw')), 0)
            patch = np.transpose(patch, (1,2,0))
            lr_patches.append(patch)

    hr_patches = np.array(hr_patches)        
    lr_patches = np.array(lr_patches)  

    np.random.seed(1004)
    np.random.shuffle(hr_patches)
    np.random.seed(1004)
    np.random.shuffle(lr_patches)
    
    return hr_patches, lr_patches

def psnr_metric(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
  return tf.image.ssim(y_true, y_pred, max_val=1.0)
