import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from subpixel_conv2d import SubpixelConv2D
from tensorflow.keras.models import Model
import model
from utils import *

seed = 15
scale = 4
num_filter = 32
layer_depth = 4  
checkpoint_path = "checkpoint"
epoch = 400
batch = 128

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.random.set_seed(seed)

#train, valid이미지 path 불러오기
image_root = pathlib.Path('./content/images')
all_images_paths=list(image_root.glob('*/*'))

train_path, valid_path = [], []

for image_path in all_images_paths:
    if str(image_path).split('\\')[-2]== 'train': 
        train_path.append(str(image_path))
    
    elif str(image_path).split('\\')[-2]=='val':
        valid_path.append(str(image_path))
    
hr_patches, lr_patches = preprocess(train_path, scale=scale)
hr_patches_val, lr_patches_val = preprocess(valid_path, scale=scale)

inputs = Input(shape=(None, None, 1), name='input')

outputs = model.unet(inputs, out_channels=1, layer_depth=4, filters_orig=32, kernel_size=3)

model = Model(inputs, outputs)

checkpoint_filepath = 'checkpoint/scale{}'.format(scale)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_psnr_metric',
    mode='max',
    save_best_only=True)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mse', metrics=[psnr_metric, ssim_metric])

history = model.fit(lr_patches,hr_patches, 
                              epochs=epoch, 
                              batch_size=128, 
                              validation_data=(lr_patches_val, hr_patches_val), 
                              verbose=1,
                               callbacks=[model_checkpoint_callback])
