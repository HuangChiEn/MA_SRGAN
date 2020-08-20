#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:51:04 2020

@author: joseph
"""
#import math
#from dataloader import DataLoader
#from skimage import color
import imageio
import numpy as np
from skimage.measure import compare_ssim
import tensorflow as tf

class evaluator: 
    '''
    def psnr(self, target, ref):
    	## Transform the image into float64
        target_data = np.array(target, dtype=np.float64)
        ref_data = np.array(ref,dtype=np.float64)
        ## differential value between 2 images
        diff = ref_data - target_data
        ## Flatten the image according to the third-channel
        diff = diff.flatten('C')
        ## calculate MSE value
        rmse = math.sqrt(np.mean(diff ** 2.))
        ## setting epsilon to prevent divide 0 error.
        eps = np.finfo(np.float64).eps
        if(rmse == 0):
            rmse = eps 
        return 20*math.log10(255.0/rmse)
    '''
    '''
    
    def cal_ssim(self, im1,im2):    
        ## Assertion of the shape and rgb convertion 
        assert(im1.shape == im2.shape)
        if len(im1.shape)>2:
            im1 = color.rgb2gray(im1)
            im2 = color.rgb2gray(im2)
        ## SSIM calculation :
        mu1 = im1.mean()
        mu2 = im2.mean()
        sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
        sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
        k1, k2, L = 0.01, 0.03, 255
        C1 = (k1*L) ** 2
        C2 = (k2*L) ** 2
        C3 = C2/2
        l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
        c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
        s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
        ssim = l12 * c12 * s12
        return ssim
    '''
    
    def psnr(self, target, ref):
        return tf.image.psnr(target, ref, max_val=255.0)
    
    def ssim(self, target, ref):
        return compare_ssim(target, ref)
    
    def imread(self, path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)
    
    
if __name__ == '__main__':
    dataset = "IOM"
    ''' Setting path '''
    eva = evaluator()
    i = 19500
    num = str(i)
    n1 = 1
    n2 = 2
    n3 = 1
    order0 = "0"
    order1 = "1"
    path = {"sr" : ("../images/" + dataset + "/" + num + "_super" + order1 + ".png"), 
            "lr" : ("../images/" + dataset + "/" + num + "_lowres" + order1 + ".png"),
            "hr" : ("../images/" + dataset + "/" + num + "_original" + order1 + ".png")
            #"hr" : ("../datasets/" + dataset + "/cls" + str(n1) + "_" + str(n2) + "_" + str(n3) + ".bmp")}
    }
    ''' read img '''
    #print(len())
    srImg = eva.imread(path["sr"])
    lrImg = eva.imread(path["lr"])
    hrImg = eva.imread(path["hr"])
    print("psnr : ", eva.psnr(srImg, hrImg))
    print("ssim : ", eva.cal_ssim(srImg, hrImg))  # hr and sr have same dim