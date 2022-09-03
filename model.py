import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Lambda, Dropout, 
                                     MaxPooling2D, LeakyReLU, concatenate, BatchNormalization, Add)
from subpixel_conv2d import SubpixelConv2D

    
def newshape(x):
    N, H, W, C = x.shape 
    # 제 1영역
    x1= x[:,0:H:2,0:W:2]

    #제 2영역
    x2= x[:,0:H:2,1:W:2]

    #제 3영역
    x3= x[:,1:H:2,0:W:2]

    #제 4영역
    x4= x[:,1:H:2,1:W:2]

    x = tf.stack([x1[:,:,:,0],x2[:,:,:,0],x3[:,:,:,0],x4[:,:,:,0]], axis=-1)
    for i in range(1,C):
        x0 = tf.stack([x1[:,:,:,i],x2[:,:,:,i],x3[:,:,:,i],x4[:,:,:,i]], axis=-1)
        x = tf.keras.layers.concatenate([x,x0], axis=-1)
    return x

def unet_conv_block(x, filters, kernel_size=3, batch_norm=True, dropout=False,
                    name_prefix="enc_", name_suffix=0):
    name_fn = lambda layer, num: '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    x = Conv2D(filters, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 1))(x) 
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 1))(x) 
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 1))(x)  

    x = Conv2D(filters/2, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 2))(x)
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 2))(x)
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 2))(x)#

    return x

    
def unet_btn_block(x, filters, kernel_size=3, batch_norm=True, dropout=False,
                    name_prefix="btn_", name_suffix=0):

    name_fn = lambda layer, num: '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    x = Conv2D(filters, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 1))(x) #kernel을 He 정규분포 초기값 설정
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 1))(x) #배치 정규화
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 1))(x)  #활성함수 LeakyReLU, 음의 기울기 계수 음수에 곱함
    

    return x

def unet_deconv_block(x, filters, kernel_size=3, batch_norm=True, dropout=False,
                      name_prefix="dec_", name_suffix=0):

    name_fn = lambda layer, num: '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    # First convolution:
    x = Conv2D(filters, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 1))(x)
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 1))(x)
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 1))(x)
    
    
    #second convoluiton
    x = Conv2D(filters, kernel_size=kernel_size, activation=None,
               kernel_initializer='he_normal', padding='same',
               name=name_fn('conv', 2))(x)
    if batch_norm:
        x = BatchNormalization(name=name_fn('bn', 2))(x)
    x = LeakyReLU(alpha=0.3, name=name_fn('act', 2))(x)

    return x


def unet(x, out_channels=1, layer_depth=4, filters_orig=32, kernel_size=4,
         batch_norm=True, final_activation='sigmoid'):

    # Encoding layers:
    filters = filters_orig
    outputs_for_skip = []
    for i in range(layer_depth): #4번 반복
        conv_block = unet_conv_block(x, filters, kernel_size,
                                     batch_norm=batch_norm, name_suffix=i) 
        outputs_for_skip.append(conv_block)
        x= newshape(conv_block)

        #활용시 kernel=32
        filters = min(filters * 2, 512) # 128, 256, 512

    # Bottleneck layers:
    x = unet_btn_block(x, filters, kernel_size, name_suffix='btleneck') #1024

    # Decoding layers:
    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig) #512, 256, 128

        x = SubpixelConv2D(upsampling_factor=2)(x)
        
        shortcut = outputs_for_skip[-(i+1)] #skip-connection
        x = tf.keras.layers.Add()([x, shortcut])
        
        x = unet_deconv_block(x, filters, kernel_size,
                                         batch_norm=batch_norm,
                                         dropout=False, name_suffix=i)

    filters = filters // 2 #채널 수 맞춰주기 위해 추가
    
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
               padding='same', name='dec_out1')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
               padding='same', name='dec_out2')(x)
    x = Conv2D(filters=out_channels, kernel_size=1, activation=final_activation,
               padding='same', name='dec_output')(x)
    return x