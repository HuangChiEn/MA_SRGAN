#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:11:19 2020

@author: joseph
"""

import numpy as np
import math 

import sys
sys.path.insert(1, '../shared_module')  
from dataloader import DataLoader  # Extandable for data preprocessing
from modelManager import ModMang
from Image_Generator_Using_Center_Score import ImageGenerator

sys.path.insert(2, '../submodule')  
from submodule import load_module
import losses_package as loss_pkg

## Notice : For passing the unit test, any information will not represesnt in command when you run the code.
if __name__ == '__main__':
    ## self-define parameters :
    tra_paras = { "batch_size":12,
            "img_shape":(120, 140, 3),
            "epochs":80
            }
    
    ## mask generator :
    msk_gen = load_module('msk_gen', 
                          {'img_shape' :  tra_paras['img_shape']})
    
    loss_dict = { 'Pupil_Bondary' : loss_pkg.IoU_Square_Loss,
      'Iris_Bondary' : loss_pkg.IoU_Square_Loss,
      'Mask' : loss_pkg.softmax_sparse_crossentropy_ignoring_last_label}
    
    msk_gen.compile(loss=loss_dict, optimizer='adam')
    
    ## prepaer training data :
    train_generator = ImageGenerator('./data/train.txt', 
                                 tra_paras['img_shape'],
                                 batch_size=tra_paras['batch_size'])
    
    train_len = train_generator.n
    msk_gen.fit_generator(
        train_generator.next_batch(),
        steps_per_epoch=math.floor( train_len // tra_paras['batch_size'] + 1 ),
        epochs=tra_paras['epochs'])
    
    msk_gen.save_weights('../pretrain/CASIA/att_dis/test.h5')
    