#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:25:38 2020

@author: k0 (first contributor) && 
               joseph(second contributor)
"""
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, AveragePooling2D, UpSampling2D
from keras.layers import Add, Multiply, Lambda, Concatenate
from keras.layers import BatchNormalization, Dropout, Activation
import keras.backend as KTF
import tensorflow as tf
from keras.utils import multi_gpu_model


''' TODO.1 : The mask operation of image will further move to ESRGAN.py module,
    and this code will be removed soon.'''
    
'''
def build_msk_dis(img_shape, img_mask_shape):
    img, mask = Input(shape=img_shape), Input(shape=img_mask_shape)
    
    gray_img = Lambda(tf.image.rgb_to_grayscale)(img)
    binarlized_prb_mask = UpSampling2D(size=(4, 4), interpolation='bilinear')(mask)
    
    ## first channel(index 0) denote front-part of img, which allows to take attention.
    prb_compare_layer = Lambda(lambda tensor : KTF.greater(tensor[:, :, :, 1], tensor[:, :, :, 0]))
    bool_mask = Lambda(KTF.expand_dims)(prb_compare_layer((binarlized_prb_mask)))
    img_mask = Lambda(lambda x : tf.dtypes.cast(x, tf.float32))(bool_mask)
    masked_img = Multiply()([gray_img, img_mask])
    
    model = Model(inputs=[img, mask], outputs=[masked_img])
    return model

'''
## Note : these code use in Given Mask-attention basedline(not DFCN).
def build_msk_dis(img_shape, img_mask_shape):
    
    def grayscale_to_rgb(images, channel_axis=-1):
        images = Lambda(lambda x: KTF.expand_dims(x, channel_axis))(images)
        tiling = [1] * 4    # 4 dimensions: Batch, H, W, C
        tiling[channel_axis] *= 3
        ten_til = KTF.constant(tiling, dtype='int32')
        images = Lambda(lambda x : KTF.tile(x, ten_til))(images)
        return images
    
    rgb_img, bool_mask = Input(shape=img_shape), Input(shape=img_mask_shape)
    rgb_mask = grayscale_to_rgb((bool_mask))
    
    masked_img = Multiply()([rgb_img, rgb_mask])
    return Model(inputs=[rgb_img, bool_mask], outputs=masked_img)


def build_msk_generator(img_shape):
    
    def Dense_Block(inp, filters, name):
        ## dense_block_part 1
        x = BatchNormalization()(inp)
        x = ReLU()(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        ## concate with 
        x = Concatenate()([x, inp])
        
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D((filters*2), (3, 3), padding='same')(x)
        
        x = Concatenate()([x, inp])
        
        x = AveragePooling2D(padding='same')(x)
        
        return x 
    

    def Decode_Block(inp, filters, pool_layer, idx):
        
        ##  DeConvolution part : 
        if (idx==0) :
            x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), 
                                padding='same',output_padding=(1,1))(inp)
            
        elif (idx==1) :
            x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), 
                                padding='same',output_padding=(0,1))(inp) 
        elif (idx==2) :
            x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), 
                                padding='same',output_padding=(1,1))(inp)
            
        elif (idx==3) :
            x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), 
                                padding='same',output_padding=(1,1))(inp)
        ## Fuse layer --
        x = Add(name=('Fuse'+str(idx)))([x, pool_layer])
        
        ## Process layer
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters*2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, (1, 1), padding='same', 
                   name=('Process_layer'+str(idx)))(x)
        
        return x
    
    
    in_tensor = Input(shape=img_shape) 
    
    ## Encoder :                      
    x = Conv2D(64, (3, 3), padding='same', name='Conv1')(in_tensor)
    ## go into dense block 1 ~ 5 --
    pool_list = []
    for i in range(0,5) :
        name = 'dense_block_' + str(i+1)
        x = Dense_Block(x,48,name)
        if ( i < 4 ) :
            pool_list.append(x)
            
    x = Conv2D(1024, (7, 7), padding='same', name='Conv6')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(2048, (1, 1), padding='same', name='Conv7')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(2, (1, 1), padding='same', name='score')(x)

    # Decoder :
    for i in range(0,4) :
        x = Decode_Block(x,(448-96*i),pool_list[3-i],i)
        
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    # mask --
    mask = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    mask = Dropout(0.3)(mask)
    mask = Conv2D(2, (1,1), padding='same', name='Mask')(mask) 
    # pupil position --
    pupil_pos = Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same')(x)
    pupil_pos = Dropout(0.5)(pupil_pos)  
    pupil_pos = Conv2D(2, (1,1), padding='same', name='Pupil_Bondary_Conv2D')(pupil_pos)
    pupil_pos = Activation('sigmoid', name='Pupil_Bondary')(pupil_pos)
    # iris position --
    iris_pos = Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same')(x)
    iris_pos = Dropout(0.5)(iris_pos) 
    iris_pos = Conv2D(2, (1,1), padding='same', name='Iris_Bondary_Conv2D')(iris_pos)
    iris_pos = Activation('sigmoid', name='Iris_Bondary')(iris_pos)

    tmpModel = Model(inputs=in_tensor, output=[pupil_pos, iris_pos, mask])
    model = multi_gpu_model(tmpModel, gpus=2)
    return model
    