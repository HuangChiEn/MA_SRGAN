#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:08:34 2020

@author: joseph
"""

"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

"""

import sys
sys.path.insert(1, '../shared_module')  
from dataloader import DataLoader  # Extandable for data preprocessing
from modelManager import ModMang

sys.path.insert(2, '../ESRGAN')  
from submodule import load_module
import losses_package as loss_pkg

import numpy as np
import imageio as imgIO

def store_img(img_lst, file_name, file_ext='png', debug_info=None, msk=False):
    for (idx, img), filNam in zip(enumerate(img_lst), file_name):
        if (~msk):
            img = 0.5 * img + 0.5
        imgIO.imwrite("../images/debug/%s_b%s_e%s.%s"% (filNam, idx, eph, file_ext), img)


if __name__ == "__main__":
    ## test interface :
    data_loader = DataLoader(data_set_name='CASIAwMskLft', 
                             hr_img_size=(480, 640), scalr=4)
    iters, batch_size = 10001, 16
    
    data_gen = data_loader.ld_data_gen(batch_size=16, fliplr=True, include_msk=True, shuffled=True, )
    for eph in range(iters):
        filNam, _, hr_imgs, lr_imgs = next(data_gen)
        imgs_msk = data_loader.load_correspond_mask()
        
        store_img(hr_imgs, filNam, debug_info=eph)
        #store_img(lr_imgs, filNam, debug_info='lr')
        #store_img(imgs_msk, filNam, debug_info='msk', msk=True)
     
        
        
    
    