from utils import *
import os
import tensorflow as tf
import numpy as np
import math
import cv2
import random
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import pathlib
import PIL.Image as pil_image
from model import *

checkpoint = "checkpoint"
scale = 2
layer_depth = 4
num_filter = 16


inputs = Input(shape=(None, None, 1), name='input')
outputs = unet(inputs, layer_depth=layer_depth, filters_orig=num_filter, kernel_size=3)
model = Model(inputs, outputs)
model.load_weights("{}/scale{}".format(checkpoint, scale))

set5_image_root = pathlib.Path('./content/Set5')
set5_image_paths = list(set5_image_root.glob('*.*'))

set14_image_root = pathlib.Path('./content/Set14')
set14_image_paths = list(set14_image_root.glob('*.*'))

bsd100_image_root = pathlib.Path('./content/bsd100')
bsd100_image_paths = list(bsd100_image_root.glob('*.*'))

urban100_image_root = pathlib.Path('./content/Urban100')
urban100_image_paths = list(urban100_image_root.glob('*.*'))

img_list = [set5_image_paths, set14_image_paths, bsd100_image_paths, urban100_image_paths]
data = ["set5", "set14", "bsd100", "urban100"]

for img_path, data_name in zip(img_list, data):
    for i,image_path in enumerate(img_path):
        our_psnr = []
        our_ssim = []
        bicubic_psnr = []
        bicubic_ssim = []
        
        hr = load_image(image_path)
        hr = modcrop(hr, scale, train=False)
        lr = generate_lr(hr, scale)

        hr = image_to_array(hr)
        hr = normalize(convert_rgb_to_ycbcr(hr.astype(np.float32)))
        crcb = hr[1:]
        hr = hr[0]

        lr = image_to_array(lr)
        lr = normalize(rgb_to_y(lr.astype(np.float32), 'chw'))

        predict_hr = model.predict(np.expand_dims(np.expand_dims(lr, axis=-1), axis=0))
        predict_hr = np.squeeze(predict_hr)

        if save_image:
            ycrcb_img = np.array([predict_hr,crcb[0],crcb[1]])
            rgb_img = np.transpose(convert_ycbcr_to_rgb(ycrcb_img),(1,2,0))
            cv2.imwrite("result/{}/{}.png".format(data_name,i), cv2.cvtColor(np.clip(rgb_img*256,0,256),cv2.COLOR_BGR2RGB))
        psnr = PSNR(hr, lr, max=1. , shave_border=scale)
        bicubic_psnr.append(psnr)
        psnr = PSNR(predict_hr, lr, max=1. , shave_border=scale)
        our_psnr.append(psnr)

        ssim = ssim(hr, lr, max=1. , shave_border=scale)
        bicubic_ssim.append(ssim)
        ssim = ssim(predict_hr, hr, max=1. , shave_border=scale)
        our_ssim.append(ssim)
        
        print("Dataset: {}".format(data_name))
        print("Ours PSNR value: {}".format(np.mean(our_psnr)))
        print("Ours SSIM value: {}".format(np.mean(our_ssim)))
        
        print("Ours PSNR value: {}".format(np.mean(bicubic_psnr)))
        print("Ours SSIM value: {}".format(np.mean(bicubic_ssim)))