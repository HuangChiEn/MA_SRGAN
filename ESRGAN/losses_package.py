#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:40:36 2020

@author: k0 (first contributor) && 
               joseph*(same contribution)
"""

import keras.backend.tensorflow_backend as KTF 
import tensorflow as tf

## For ESRGAN losses :

## Relativistic Leaste squre Average GAN loss function : 
# Define generator and discriminator losses according to RaGAN described in Jolicoeur-Martineau (2018).
# Dummy predictions and trues are needed in Keras. 

## RLaGAN - loss  <more stable training process>
def custom_rela_dis_loss(dis_real, dis_fake):
    
    def rel_dis_loss(dummy_pred, dummy_true):
        ## Critic term ( output before activation ) 
        real_diff = dis_real - KTF.mean(dis_fake, axis=0)
        fake_diff = dis_fake - KTF.mean(dis_real, axis=0)

        return KTF.mean(KTF.pow(real_diff-1,2),axis=0)+\
                KTF.mean(KTF.pow(fake_diff+1,2),axis=0)
                 
    return rel_dis_loss

def custom_rela_gen_loss(dis_real, dis_fake):
    
    def rel_gen_loss(dummy_pred, dummy_true):
        ## Critic term ( output before activation ) 
        real_diff = dis_real - KTF.mean(dis_fake, axis=0)
        fake_diff = dis_fake - KTF.mean(dis_real, axis=0)
        
        return KTF.mean(KTF.pow(fake_diff-1,2),axis=0)+\
                KTF.mean(KTF.pow(real_diff+1,2),axis=0)
            
    return rel_gen_loss

# perceptual loss
def perceptual_loss(fake_fea, real_fea):
    percept_loss = tf.losses.mean_squared_error(fake_fea, real_fea)
    return percept_loss

# pixel loss
def pixel_loss(fake_hr, img_hr):
    pixel_loss = tf.losses.absolute_difference(fake_hr, img_hr) 
    return pixel_loss

## masked attention loss
def attention_l1_loss(att_fake_img, att_hr_img):
    atten_loss = tf.losses.absolute_difference(att_fake_img, att_hr_img) 
    return atten_loss


## For DFCN losses :

def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = KTF.reshape(y_pred, (-1, KTF.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = KTF.one_hot(tf.to_int32(KTF.flatten(y_true)), KTF.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -KTF.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = KTF.mean(cross_entropy)

    return cross_entropy_mean


def IoU_Square_Loss(y_true, y_pred):
    
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
        
        return 1 - (numerator + 1) / (denominator + 1)
    
    
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))
        
        alpha=0.25
        gamma=2
        
        weight_a = alpha * (1 - y_pred) ** gamma * y_true
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - y_true)
        loss = (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 
        
        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)
    
    """
            Input :
            y_true[batch,w,h,3] : score, distance, radius
            y_pred[batch,w,h,2] : score, radius . And radius should rescale
    """
    
    radius_rescale = 240   ## 120
    tensor_center = y_true[:,:,:,0]
    
    min_radius = tf.minimum( y_true[:,:,:,2], y_pred[:,:,:,1]*radius_rescale )
    max_radius = tf.maximum( y_true[:,:,:,2], y_pred[:,:,:,1]*radius_rescale )
    
    tensor_union = ( max_radius*max_radius )
    tensor_intersection = (min_radius * min_radius)
    
    tensor_intersection = tf.clip_by_value(tensor_intersection, KTF.epsilon(), tensor_intersection)
    
    
    tensor_log_IoU = -tensor_center*tf.log(tensor_intersection/tensor_union)
    
    iou = (tf.reduce_sum(tensor_log_IoU) / 
                 tf.clip_by_value(tf.reduce_sum(tensor_center), KTF.epsilon(), 
                 tf.reduce_sum(tensor_center)))
    dice = dice_loss(y_true[:,:,:,0],y_pred[:,:,:,0])
    focal = focal_loss(y_true[:,:,:,0],y_pred[:,:,:,0])
    return iou + dice + focal

