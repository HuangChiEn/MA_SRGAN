#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import  ## execute the following code in python3.x enviroment
"""
Created on Tue May  18 15:32:00 2020
@author: Josef-Huang

@@ The following code is stable ' V 2.2 (stable) version '
Signature :
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@                                                                  @@
@@    JJJJJJJJJ    OOOO     SSSSSSS   EEEEEE   PPPPPPP   HH    HH   @@
@@       JJ       O    O    SSS       E        PP   PP   HH    HH   @@
@@       JJ      O      O    SSS      EEEEEE   PPPPPPP   HHHHHHHH   @@
@@       JJ       O    O         SS   E        PP        HH    HH   @@
@@     JJJ         OOOO     SSSSSSS   EEEEEE   PP        HH    HH   @@
@@                                                                  @@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    

Description :
    The following code is the implementation of super resolution GAN by keras mainly with tensorflow backend
    engining.
    
    The kernel module be divided into  generator, discriminator, auxiliary_featuretor. The different module 
    can load different type of basic block :  
    (1) generator -> RRDRB(Residual in Residual Dense Residual Block)
        -> The subpixel conv are eliminated(currently), due to their effect are not good~(see problem sample)
        -> However, respect with paper proposed model, I trace back to ECPNetwork.
        (main stream)@->
            @-> I use keras Upsampling2D function to upsampling(nearest), 
                I may attempt to replace it into TransposeConv or SubPixel Conv.
            @-> Due to the network size, I should reduce part of it to implement more
                functionality and accelerate the prediction phase.
                @@-> The 2014 proposed VGG-net should be replace to other structure.
                    The mobile-net v2, v3 are candidate of original VGG.
            @-> The attention mechnism will be include into ESRGAN,
                for generate the data of specific domain task. 
                    
Acknowledegment :
    The following code are referenced from a lot of released source code and unreleased code.
    I want to greatly thank for the author on github who released the srgan.py source code.
    
    When I move the framework of released code into my project, my senior k0 Young, who offer 
    some scheme to build the parallel GPU env, so that I can run the experiments quickly.
    Beside, he found the bug in Resnet code to help me reduce the computation cost.
    
    And my senior sero offer a lot of advice to help me de-very-big-bug with lower cost.
    The learning rate should adjust even in Adam method.
    
    At the result, of course my prof. Liu, who offer the lab to support our research.
    Every research can not be done completely without the env support!!
    
Notice : 
    As the file name, the edsr gan module will be implement in this code.
    The RaGAN loss function will be implement in the code, and the loss weight coefficient 
    with different loss function still request to decide.
    At the present, [1.0 (feature content_loss), 1e-3 (RaDis loss), 1e-3 (L1 norm loss)] loss 
    weight are in use.
    
    At the next stage, i'm going to change the G, auxMod structure with transConv2d and MobileNet, 
    instead of bilinear upsampling method and VGG19.

            Josef-Huang...2020/06/04 (Donnerstag)
"""

## tensorflow backend--GPU env seeting : 
from keras.utils import multi_gpu_model
import os
import tensorflow as tf
import keras.backend as KTF

## some related utility : 
import imageio as imgIO
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../shared_module')

## keras package : the high-level module for deep learning 
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam

## setting callback mechnism : 
'''
# 1). FREEUSE
from keras.callbacks import TensorBoard
ensure_dir( './TensorBoard/' + file_name )
path = './TensorBoard/' + file_name + 'logs'  
tensorboard = TensorBoard(log_dir=path, 
                          histogram_freq=0,
                          batch_size=batch_size, 
                          write_graph=True, 
                          write_grads=True, 
                          write_images=True )

# 2). FREEUSE
from keras.callbacks import EarlyStopping
## Early stopping technique : 
##  Not very recommand to use, since the generator may request more epochs to
##      generate high-quilty image in sometimes. 
##      if you want to set early stopping, at least 10000 epochs.

earlyStop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=500, 
                          verbose=1 , 
                          mode='auto')
'''
##TODO.2 : Adding the checkpoints mechnism.. 

## Self-define package 
from dataloader import DataLoader  # Extandable for data preprocessing
from modelManager import ModMang
from submodule import load_module
from Image_Generator_Using_Center_Score import ImageGenerator
import SrGAN    
#from keras.layers import UpSampling2D, Lambda    # for pretraining RRDRB structure.
import losses_package as loss_pkg

class ESRGAN():
    
    def __init__(self, lr_shape, img_scalar,  
                 n_G_filt, n_RRDRB, n_DRB, res_scalar, n_D_filt,  
                 n_gpus,  exe_mode, pre_model_dir, pretrain,  
                 loss_weights, learn_rate, 
                 **_):  

        ## Image input structure : 
        self.lr_height, self.lr_width, self.channels = self.lr_shape = lr_shape
        self.img_scalar = img_scalar
        self.hr_height, self.hr_width, _ = [dim*img_scalar for dim in lr_shape]
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        self.n_gpus = n_gpus
        self.model_man =  ModMang(save_direc=pre_model_dir)
        
        ## Training Model Seeting : 
        if (exe_mode == "training"):
            # Common Factor : 
            self.loss_weights = loss_weights
            self.optimizer = Adam(lr=learn_rate, beta_1=0.5, beta_2=0.999, amsgrad=True)

            ## (1) Load Basic Network Components : 
            
            ## load generator - 
            #   parameter setting 
            G_params = {
                    ## self-define :
                    'num_of_filts' : n_G_filt,
                    'num_of_RRDRB' : n_RRDRB,
                    'lr_shape' : self.lr_shape,
                    ## default :
                    'num_of_DRB' : n_DRB,
                    'upScalar' : self.img_scalar,
                    'resScal' : res_scalar
                    }
            self.generator = load_module("generator", G_params)
            if pretrain == True:
                srgan = SrGAN.SrGAN(RRDB=self.generator)
                srgan.pretrain()
                self.generator = self.model_man.simple_load(RRDB=self.generator)
            
            ## load feature_discriminator - 
            self.feature_dis = load_module("feature_dis", {'hr_shape' : self.hr_shape})
            self.feature_dis.compile(loss="mse", optimizer=self.optimizer)
            
            ## load discriminator - 
            D_params = {'num_of_filts' : n_D_filt,
                    'hr_shape' : self.hr_shape}
            self.discriminator = load_module("discriminator", D_params)
            
            ## load attention-based (mask generator) and (mask operator) - 
            #self.msk_shape = (self.lr_height, self.lr_width, 2)  ## mask : 0, background, 1 iris.
            self.msk_shape = (self.hr_height, self.hr_width)
            
            # mask generator :
            Msk_gen_params = {'img_shape' :  self.lr_shape}
            self.msk_gen = load_module("msk_gen", Msk_gen_params)
            
            loss_dict = { 'Pupil_Bondary' : loss_pkg.IoU_Square_Loss,
              'Iris_Bondary' : loss_pkg.IoU_Square_Loss,
              'Mask' : loss_pkg.softmax_sparse_crossentropy_ignoring_last_label}
            
            self.msk_gen.compile(loss=loss_dict, optimizer="adam")
            
            # mask operator :
            Attdis_params = {'img_shape' : self.hr_shape,
                    'img_mask_shape' : self.msk_shape} 
            self.atten_dis = load_module("attention_dis", Attdis_params)
#            
            
            ## (2) Build high-level inner modules : 
            
            ## build RaGAN -  by previous defined basic network component.
            self.RaDis = self.__inner_build_RaGAN()
            
            ## build SrGAN - (main module, need compile)
            self.SrGAN = self.__inner_build_SrGAN()
            
        elif(exe_mode == "predicting"):
            ## Configure model manager :
            G_params = {
                    ## self-define :
                    'num_of_filts' : n_G_filt,
                    'num_of_RRDRB' : n_RRDRB,
                    'lr_shape' : self.lr_shape,
                    ## default :
                    'num_of_DRB' : n_DRB,
                    'upScalar' : self.img_scalar,
                    'resScal' : res_scalar
                    }
            self.generator = load_module("generator", G_params)
            self.generator = self.model_man.load_model_weight(generator=self.generator)
            
        else:
            raise ValueError("Such mode do not exist, the module only support \
                             training or predicting in initialization stage.\n")

    def __inner_build_RaGAN(self):
        self.generator.trainable = False
        self.discriminator.trainable = True
        
        ## define discriminator judge part :
        img_lr, img_hr = Input(shape=self.lr_shape), Input(shape=self.hr_shape)
        fake_hr = self.generator(img_lr)
        dis_real, dis_fake = self.discriminator(img_hr), self.discriminator(fake_hr)
        relative_dis_loss = loss_pkg.custom_rela_dis_loss(dis_real, dis_fake)
        model = Model(inputs=[img_lr, img_hr], outputs=[dis_real, dis_fake])
        model.compile(optimizer=self.optimizer, loss=[relative_dis_loss, None], loss_weights=[1.0, 0])
        return model
    
        
    def __inner_build_SrGAN(self):
        ## Trainable setting for properly update G and D parameters : 
        self.generator.trainable = True        ##  Only train G during the SrGAN.train_on_batch phase
        self.discriminator.trainable = False
        self.feature_dis.trainable = False
        self.atten_dis.trainable = False
        
        ## Image setting for model input structure : 
        img_lr, img_hr, img_mask = Input(shape=self.lr_shape), Input(shape=self.hr_shape), Input(shape=self.msk_shape)
        #img_lr, img_hr = Input(shape=self.lr_shape), Input(shape=self.hr_shape)
        
        fake_hr = self.generator(img_lr)
        
        ## Extract the feature of mask ROI.
        ROI_part = self.atten_dis([fake_hr, img_mask])
        ROI_fea = self.feature_dis(ROI_part)
        
        ##  Discriminator determines validity of generated high resolution images 
        dis_real, dis_fake = self.discriminator(img_hr), self.discriminator(fake_hr)
        relative_gen_loss = loss_pkg.custom_rela_gen_loss(dis_real, dis_fake)
        
        ## At the result : build SrGAN with multi-output and loss 
        tmpModel = Model(inputs=[img_lr, img_hr, img_mask], outputs=[fake_hr, ROI_fea, dis_real, dis_fake], name='SrGAN')
        #tmpModel = Model(inputs=[img_lr, img_hr], outputs=[fake_hr, fake_features, dis_real, dis_fake], name='SrGAN')   
        
        model = multi_gpu_model(tmpModel, gpus=self.n_gpus)
        ## TODO: Adding loss dictionary.
        model.compile(optimizer=self.optimizer, \
                loss=[loss_pkg.pixel_loss, loss_pkg.perceptual_loss, relative_gen_loss, None], \
                loss_weights=[self.loss_weights['pixel'], self.loss_weights['att_percept'], self.loss_weights['gen'], 0.])
        '''
        model.compile(optimizer=self.optimizer, \
                loss=[loss_pkg.pixel_loss, loss_pkg.perceptual_loss, relative_gen_loss, None], \
                loss_weights=[self.loss_weights['pixel'], self.loss_weights['percept'], self.loss_weights['gen'], 0.])
        '''
        return model
    
    def training(self, data_set_name, load_G_D_weight, load_msk_gen_weight, 
                 batch_size, epochs, sample_interval, save_dir_name, save_generator, 
                 **_): 
        ## Save ROI part
        def sample_images(epoch, save_dir_name):
            os.makedirs('../images/%s' % save_dir_name, exist_ok=True)
            r, c = 2, 2
            imgs_hr, imgs_lr, imgs = self.data_loader.ld_data_rnd(batch_size=2)
            fake_hr = self.generator.predict(imgs_lr)
    
            # Rescale images 0 - 1
            imgs_lr = 0.5 * imgs_lr + 0.5
            fake_hr = 0.5 * fake_hr + 0.5
            imgs_hr = 0.5 * imgs_hr + 0.5
            imgs = 0.5 * imgs + 0.5
    
            # Save generated images and the high resolution originals
            titles = ['Generated', 'Original']
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for row in range(r):
                for col, image in enumerate([fake_hr, imgs_hr]):
                    axs[row, col].imshow(image[row])
                    axs[row, col].set_title(titles[col])
                    axs[row, col].axis('off')
                cnt += 1
            fig.savefig("../images/%s/%d.png" % (save_dir_name, epoch))
            plt.close()
            
            '''Due to check the img size and offer the comparison between sr, hr and lr img,
                   we use the savefig function of matplotlib package to save the image. '''
            for idx in range(r):
                ## Setting the figure(s) 
                fig0 = plt.figure(num=0)
                fig1 = plt.figure(num=1)
                fig2 = plt.figure(num=2)
                
                ## Plot on each figure :
                
                ## plot lower resolution image
                plt.figure(fig0.number)
                plt.imshow(imgs_lr[idx])
                
                ## plot super resolution image
                plt.figure(fig1.number)
                plt.imshow(fake_hr[idx])
                
                ## plot high resolution image (original image)
                plt.figure(fig2.number)
                plt.imshow(imgs[idx])
                
                ## Saving the image in each figure 
                fig0.savefig('../images/%s/%d_lowres%d.png' % (save_dir_name, epoch, idx))
                fig1.savefig('../images/%s/%d_super%d.png' % (save_dir_name, epoch, idx))
                fig2.savefig('../images/%s/%d_original%d.png' % (save_dir_name, epoch, idx))
                
                # Close the figure.
                plt.close(fig0)
                plt.close(fig1)
                plt.close(fig2)
        
        self.data_set_name = data_set_name
        self.data_loader = DataLoader(data_set_name=self.data_set_name,
                                      hr_img_size=(self.hr_height, self.hr_width), 
                                      scalr=self.img_scalar)
        
        self.record_his = {  ## Recordr the history of each loss into the list.
                'RaDis':[], 'SrGAN':[], 'G':[],
                'FeaDis':[], 'AttDis':[], 'RaGen':[]
                }
        
        ## load G and D model weights : 
        if load_G_D_weight:
            self.generator = self.model_man.load_model_weight(generator=self.generator)
        ## see TODO.2 : add patch-GAN discriminator in training phase.
        
        ## PLUGIN_DONE : directly load the module weight. TODO : add to modMan..
        if load_msk_gen_weight:
            self.msk_gen.load_weights('../pretrain/CASIA/att_dis/CA_exp001_0.h5')  
        else:
            train_generator = ImageGenerator('./data/train.txt', 
                                 self.lr_height, self.lr_width, self.channels,
                                 batch_size=batch_size)
            train_len = train_generator.n
            self.msk_gen.fit_generator(
                train_generator.next_batch(),
                steps_per_epoch=math.floor( train_len // batch_size + 1 ),
                epochs=80)
            self.msk_gen.save_weights('../pretrain/CASIA/att_dis/test.h5')
        print('Procedure of preparing the attention module is complete..\n')
        
        dummy = np.zeros((batch_size, 1), dtype=np.float32)
        start_time = datetime.datetime.now()
        
        for epoch in range(epochs):
            ## Generator training phase : 
            imgs_hr, imgs_lr, _ = self.data_loader.ld_data_rnd(batch_size, fliplr=True, include_msk=True)  ## False
            
            imgs_msk = self.data_loader.load_correspond_mask()
            #_, _, imgs_msk = self.msk_gen.predict(imgs_lr)
            
            ROI_part = self.atten_dis.predict([imgs_hr, imgs_msk])
        
            real_ROI_fea = self.feature_dis.predict(ROI_part)
        
            self.generator.trainable = True
            self.discriminator.trainable = False
            g_loss = self.SrGAN.train_on_batch([imgs_lr, imgs_hr, imgs_msk], [imgs_hr, real_ROI_fea, dummy])
            #g_loss = self.SrGAN.train_on_batch([imgs_lr, imgs_hr, imgs_msk], [imgs_hr, att_fea, dummy])

            
            ## Train RaGAN (discriminator) : 
            imgs_hr, imgs_lr, _ = self.data_loader.ld_data_rnd(batch_size, fliplr=True)
            
            self.generator.trainable = False
            self.discriminator.trainable = True
            d_loss = self.RaDis.train_on_batch([imgs_lr, imgs_hr], [dummy])
            
            elapsed_time = datetime.datetime.now() - start_time
            
            print ("Epoch : %d ; time: %s\n" % (epoch, elapsed_time))
            
            ## Evaluation model : 
            print("  RaDis out : \n  D -> {}\n\n".format(d_loss[1]))
            print("  SrGAN out : Total loss -> {}\n".format(g_loss[0]))
            print("    G -> {} ; RaGen -> {}\n".format(g_loss[1], g_loss[3]))
            print("    FeaDis -> {}\n".format(g_loss[2]))
            print("\n\n--------------------------------------\n\n")
            
            self.record_his['RaDis'].append(d_loss[1])
            self.record_his['SrGAN'].append(g_loss[0])
            self.record_his['G'].append(g_loss[1])
            self.record_his['FeaDis'].append(g_loss[2])
            self.record_his['RaGen'].append(g_loss[3])
            
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                sample_images(epoch, save_dir_name)
                
        if save_generator:
            self.model_man.save_model_weight(generator=self.generator)
            
            
    def generating_img(self, lr_img_dir, n_sr_img, gen_batch_size, file_ext):
        def store_img(img, file_name, file_ext='png'):
            # Rescale images to range [0 - 1]
            img = 0.5 * img + 0.5
            imgIO.imwrite("../images/%s/%s.%s"% (lr_img_dir, file_name, file_ext), img)
        
        self.data_loader = DataLoader(data_set_name=lr_img_dir,
                                      hr_img_size=(self.hr_height, self.hr_width), 
                                      scalr=self.img_scalar)
            
        div, mod = divmod(n_sr_img, gen_batch_size)
        assert((mod == 0))  ## Total image should be divided by batch_size exactly
        gen_epoch = div
        
        ## Get the image generator.
        img_gen = self.data_loader.get_img_generator(gen_batch_size, file_ext=file_ext)
        idx = 0 
        try:
            for _ in range(gen_epoch):
                ## HACKME0.1 : read the lower resolution image by batch
                lr_img, file_name = next(img_gen)
                sr_img = self.generator.predict(lr_img)
                for _, img in enumerate(sr_img):
                    store_img(img, file_name, file_ext)
                del lr_img
                del sr_img
                idx+=1
                
        except StopIteration:
            print("The image generating procedure are already complete.\n",\
                      "If the time of complete is shorter than you thought, ",\
                          "you may check the number of data in lower resolution datasets.\n")
       
    def lookback_hist(self, interval=100, exp='exp'):
        def plot_and_save(begin, end, del_ylim):
            epo_lst = [ x for x in range(end-begin)]
            
            fig = plt.figure()
            plt.figure(fig.number)
            plt.plot(epo_lst, self.record_his['RaDis'][begin:end], '--', color='b')
            plt.plot(epo_lst, self.record_his['SrGAN'][begin:end], '-', color='r')
            plt.plot(epo_lst, self.record_his['G'][begin:end], color='g')
            plt.plot(epo_lst, self.record_his['FeaDis'][begin:end], color='k')
            plt.plot(epo_lst, self.record_his['RaGen'][begin:end], color='y')
            
            plt.title('training history : %s-%s'%(begin, end-1))
            plt.xlabel("epochs")
            plt.ylabel("loss value")
            plt.grid(True)
            
            fig.savefig('../monitor/training_plot/CASIA/%s/epo_%s_%s.jpg'%(exp, begin, end-1))
            plt.close()
            
        ## Get the history record :
        x = 1
        [div, mod] = divmod(len(self.record_his['RaDis']), interval);
        for idx in range(div):
            for _ in range(interval): x*=0.9999
            plot_and_save((interval*idx), (interval*(idx+1)), x)
        
        
       
## Notice : For passing the unit test, any information will not represesnt in command when you run the code.
if __name__ == '__main__':
    pass    