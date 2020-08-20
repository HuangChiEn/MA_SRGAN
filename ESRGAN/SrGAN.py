#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import  ## execute the following code in python3.x enviroment

"""
Created on Tue Feb  18 15:32:00 2020
@author: Josef-Huang

@@ The following code is unstable ' V 0.4 '
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
    (1) generator -> RRDB(Residual in Residual Dense block)
        -> The subpixel conv are eliminated(currently), due to their effect are not good~(see problem sample)
        -> However, respect with paper proposed model, I trace back to ECPNetwork, that use tanh as activation.
        (main stream)@->
            @-> I use keras Upsampling2D function to upsampling(nearest), but replace the generator structure.
                the effect are as good as original structure, but better brightness.
    (2) discriminator -> RaGAN(Relativistic GAN), Conv block.
    
        (main stream)@->
            @-> I'm trying to build the RaGAN for recover more detail texture .
            
    (3) auxiliary_featuretor -> vgg19 before, vgg19 after.
        (main stream)@->
            @-> I'm trying to extract the before activated feature.
    
Acknowledegment :
    The following code are referenced from a lot of released source code and unreleased code.
    I want to greatly thank for the author on github who released the srgan.py source code, 
    and my senior k0 who offer some scheme to confirm the re-use ability, log-mechnism, 
    exception handler, and parallel GPU env to make the process of training be quickly!!
    
    At the result, of course my prof. Liu, who offer the lab to support our research.
    Every research can not be done completely without the env support!!
    
Notice : 
    As the file name, the edsr gan module will be implement in this code.
    The RRDB( Residual in Residual Dense Block ) has already be add in generator.

            Josef-Huang...2020/02/22(Donnerstag)
"""

""" tensorflow backend--GPU env seeting : """
import tensorflow as tf
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as KTF  
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.system('echo $CUDA_VISIBLE_DEVICES' )

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   ## Avoid to run out of all memory, allocate it depend on the requiring
sess = tf.Session(config=config)

KTF.set_session(sess)

'''keras package : the high-level module for deep learning '''
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.optimizers import Adam
import keras.backend as KTF ## for self-define loss function
from keras.applications import VGG19  ## keras pretraining model : for extract activated feature

'''other package'''
import datetime
import numpy as np
import argparse  ## User-define parameter setting
import sys       
sys.path.insert(1, '../sharedMod') ## Grabe the path of shared-module 

''' self-define package '''
from dataloader import DataLoader
from modelManager import ModMang

class SrGAN():
    ## define argparser
    def hard_code_parameters(self):
        '''argparse stage : the parameters of model are defined by following code'''
        parser = argparse.ArgumentParser(description="None")
        
        '''Image specification : '''
        parser.add_argument('--dataSetNam', type=str, default='IOM', help="Data set name(further maybe divide into training and testing dataset)")
        parser.add_argument('--chanl', type=int, default=3, help="The channels of input image (ie.RGB).")
        parser.add_argument('--lrImgHei', type=int, default=120, help="The height of lower resoultion input image (pixel value).")  
        parser.add_argument('--lrImgWid', type=int, default=160, help="The width of lower resoultion input image (pixel value).")
        parser.add_argument('--imgScal', type=int, default=4, help="The size of image scalar (from low resolution to super resolution).")
        
        '''Model specification : '''
        parser.add_argument('--DFilt', type=int, default=64, help="The number of filter in Discriminator.")
        parser.add_argument('--DPatSiz', type=int, default=4, help="The patch size of discriminator (see comment patch GAN).")
        
        return parser.parse_args() ## parser the arguments, get parameter via arg.parameter
        
    
    def __init__(self, RRDB):
        args = self.hard_code_parameters()
        ## Input structure :
        self.channels = args.chanl
        self.lr_height = args.lrImgHei       # Low resolution height
        self.lr_width = args.lrImgWid        # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*args.imgScal   # High resolution height
        self.hr_width = self.lr_width*args.imgScal     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.imgScal = args.imgScal
        
        ## Configure data loader :
        self.dataset_name = args.dataSetNam
        self.data_loader = DataLoader(data_set_name=args.dataSetNam,
                                      hr_img_size=(self.hr_height, self.hr_width), scalr=args.imgScal)
        ## Configure model manager :
        self.model_man =  ModMang(save_direc='../pretrain')
        
        ### Model setting ###
        optimizer = Adam(0.0002, 0.5) ## with best setting, do not change will be better
        self.generator = RRDB
        ## Auxiliry_Model structure: 
            #   We use a pre-trained VGG19 model to extract image features from the high resolution
            #   and the generated high resolution images and minimize the mse between them
        self.auxMod = self.build_auxiliary_model()
        self.auxMod.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.auxMod.trainable = False
        
        ## Discriminator structure :  
        ''' Calculate output shape of D (PatchGAN) 
            For each patch have a lot of true/false value
            So the discriminator output value according 
            to each pixel state''' 
        patch_hei = int(self.hr_height / (2**4))
        patch_wid = int(self.hr_width / (2**4))
        self.disc_patch = (patch_hei, patch_wid, 1)
            #   Build and compile the discriminator
        self.df = args.DFilt
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        ## Model relation stucture : 
            #   High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

            #   Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

            #   Extract image features of the generated img
        fake_features = self.auxMod(fake_hr)

            #   For the combined model we will only train the generator
        self.discriminator.trainable = False

            #   Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)
        
        ## Build the combined model to combind 
        ##  the generator, discriminator and auxiliary_model
        """ For parallel model : """
        mod = Model([img_lr, img_hr], [validity, fake_features])
        self.combined = multi_gpu_model(mod, gpus=4)  ## for parallel seeting
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)
        """ load G and D model weights : """
        #self.generator, self.discriminator = self.model_man.simple_load(generator=self.generator, discriminator=self.discriminator)
            
        
    ''' The following code describe the inner_structure of model '''
    def build_auxiliary_model(self, modTyp=None, fMap='after'):
        '''---------------<Inner structure modification : >-----------------'''
        oriVGG = VGG19(weights="imagenet", include_top=False)
        VGG_out = Conv2D(512, (3, 3),
                      padding='same',
                      name='block5_conv4')(oriVGG.get_layer('block5_conv3').output)
        ## Note : the block5_conv4 do not load the imagenet weight!!
        VGGBef= Model(inputs=oriVGG.input, outputs=VGG_out)
        '''-----------------------------------------------------------------'''
        
        VGGBef.outputs = [VGGBef.layers[-1].output]
        img = Input(shape=self.hr_shape)
        img_features = VGGBef(img)
        
        return Model(img, img_features)
    
    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True, kerSiz=3):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=kerSiz, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2, kerSiz=4)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2, kerSiz=4)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2, kerSiz=4)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2, kerSiz=4)
        
        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        return Model(d0, validity)
        
    ''' The following code describe the action about model '''
    def pretrain(self, epochs=1000, batch_size=4):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr, _ = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)
            ''' discriminator path size with batch size (look patch GAN comment)'''
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr, _ = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.auxMod.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the tranning progress
            print ("%d time: %s" % (epoch, elapsed_time))
            print("D loss : {} ; G loss : {} ".format(d_loss, g_loss))
        
        self.model_man.save_RRDB(RRDB=self.generator, save_part='weight')
    
'''The SrGAN module is for pretaining RRDB generator..'''
if __name__ == '__main__':
    main()
    