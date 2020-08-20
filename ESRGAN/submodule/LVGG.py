#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:45:01 2020

@author: joseph
"""

from keras.layers import Input, Conv2D, Dense, Lambda
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import keras.backend as KTF

'''
def build_LVGG(input_shape=None, blk_typ_lst=[2, 2, 3], 
         init_nfilt=64, ker_siz=(3, 3), dense_neur=4096):
    
    def ConvMax_blk(n_conv, filt_num, ker_siz, layIdx, x):
        for blkIdx in range(1, n_conv+1):
            x = Conv2D(filt_num, ker_siz,
                              activation='relu',
                              padding='same',
                              name="block{}_conv{}".format(layIdx, blkIdx))(x)
            
        x = MaxPooling2D((2, 2), strides=(2, 2), 
                                  name="block{}_pool".format(layIdx))(x)
        return x
    
    x = Input(input_shape)
    for layIdx, n_conv in enumerate(blk_typ_lst):
        x = ConvMax_blk(n_conv, init_nfilt*layIdx, ker_siz, layIdx, x)
        
    x_fea = GlobalAveragePooling2D()(x)
    x_fea = Dense(dense_neur, activation='relu', name='fc1')(x_fea)
    ## triplet loss output constraint..
    nor_fea = Lambda(lambda  x: KTF.l2_normalize(x,axis=1), name='t_emb_1_l2norm')(x_fea)
    
    return Model(inputs=x, outputs=nor_fea, name='lvgg') 
    
'''

def build_LVGG(input_shape=None, blk_typ_lst=[2, 2, 3], 
         init_nfilt=64, ker_siz=(3, 3), dense_neur=1024, cls_num=1000):
    
    def ConvMax_blk(n_conv, filt_num, ker_siz, layIdx, x):
        
        for blkIdx in range(1, n_conv+1):
            x = Conv2D(filt_num, ker_siz,
                              activation='relu',
                              padding='same',
                              name="block{}_conv{}".format(layIdx, blkIdx))(x)
    
        x = MaxPooling2D((2, 2), strides=(2, 2), 
                                  name="block{}_pool".format(layIdx))(x)
        return x
    
    in_tensor = Input(input_shape)
    x = in_tensor
    for layIdx, n_conv in enumerate(blk_typ_lst, 1):
        x = ConvMax_blk(n_conv, init_nfilt*layIdx, ker_siz, layIdx, x)
        #in_tensor = out_tensor
    
    x_fea = GlobalAveragePooling2D()(x)
    print('flag3')
    x_fea = Dense(1024, activation='relu', name='fc1')(x_fea)
    print('flag4')
    x_cls = Dense(cls_num, activation='softmax', name='fc2')(x_fea)
    print('flag5')
    return Model(inputs=in_tensor, outputs=x_cls, name='lvgg')


