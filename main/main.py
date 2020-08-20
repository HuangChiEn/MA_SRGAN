#!/usr/bin/env python3
# -*- coding: utf-8 -*-   ## In the case to print chinese character.
## execute the following code in python3.x enviroment, part of python2.x code is not allowed.
from __future__ import print_function, absolute_import  

"""
Created on Feb  28  2020
@author: Josef-Huang

@@ The following code is stable ' V 2.0 (stable version) '

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
    The following code is the implementation of super resolution GAN by keras mainly with 
    tensorflow backend engining.
    And all the assignments of model (learning, generating, etc.) can be declared in here.
            
Acknowledegment :
    The following code are referenced from a lot of released source code and unreleased code.
    I want to greatly thank for the author on github who released the srgan.py source code.
    
    When I move the framework of released code into my project, my senior k0 Young, who offer 
    some scheme to build the parallel GPU env, so that I can run the experiments quickly.
    Beside, he found the bug in Resnet code to help me reduce the computation cost.
    
    And my senior sero offer a lot of advice to help me debug with slightly painful.
    The learning rate should "self-adjust" even the Adam optimizer in used.
    
    At the result, of course my prof. Liu, who offer the lab to support our research.
    Every research can not be done completely without the env support!!
    
Notice : 
    As the file name, the edsr gan module will be implement in this code.
    However, some potential bug in RaGAN still remain for further fixed.

Copy Right : 
    Except for author name representation, all right will be released as source code.
              Josef-Huang...2020/06/12 (Freitag)
"""

## for import tensorflow backend--GPU env seeting : 
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF  
import os

## some related utility : 
import argparse  # User-define parameter setting
import sys       # Grabe the path of modules
sys.path.insert(1, '../shared_module')  
sys.path.insert(2, '../ESRGAN') 

## self-define package : 
from ESRGAN import ESRGAN

## Main function : ( Declare your parameter setting and model assignments)  
def main():
    ## (1) Parameter setting stage : the parameters used in module are defined by following code : 
    parser = argparse.ArgumentParser(description="The following user defined parameters are recommand to adjustment according to your requests.")

    ## Execution enviroment setting : 
    parser.add_argument('--cuda', type=str, default="0, 1, 2, 3", help='a list of gpus.')
    parser.add_argument('--force_cpu', type=bool, default=False, help='Force cpu execute the assignments of model whatever gpu assistance.')
    
    ## Image specification setting : 
    parser.add_argument('--lr_height', type=int, default=120, help="The height of lower resoultion input image (pixel value).")  
    parser.add_argument('--lr_width', type=int, default=160, help="The width of lower resoultion input image (pixel value).")
    parser.add_argument('--img_scalar', type=int, default=4, help="The size of image scalar (from low resolution to super resolution).")
    
    ## Model specification :  ## n_G_filt : 12, 24, 40 for dense block.
    parser.add_argument('--n_RRDRB', type=int, default=3, help="The number of residual in residual dense block(RRDB) in the part of Generator.")
    parser.add_argument('--n_G_filt', type=int, default=32, help="The number of filter in Generator.")
    parser.add_argument('--n_D_filt', type=int, default=32, help="The number of filter in Discriminator.")
    parser.add_argument('--pretrain_RRDRB', type=bool, default=False, help="The RRDB model will be pretrained and saved into the /pretrain dictorary.")
    ## HACKME.1 : The Patch Discriminator is offline, you can do better.
    #parser.add_argument('--DPatSiz', type=int, default=8, help="The patch size of discriminator (see comment patch GAN).")
    
    ## Learning setting : 
    parser.add_argument('--exe_mode', type=str, default="training", help='execution mode')
    parser.add_argument('--train_set_name', type=str, default='traSet', help="The name of training data set.")
    parser.add_argument('--epochs', type=int, default=5001, help="The number of epoch during training.")
    parser.add_argument('--batch_size', type=int, default=6, help="The number of batch size during training.")
    parser.add_argument('--samp_img_intval', type=int, default=100, help="The interval of sampling the image during training epoch.")
    parser.add_argument('--load_G_D_weight', type=bool, default=False, help="Load the weights of Generator and Discriminator to continue previous training.")
    parser.add_argument('--save_generator', type=bool, default=False, help="Save the generator to generate super resolution image in image generation procedure.")
    parser.add_argument('--load_msk_gen_weight', type=bool, default=True, help="Load the pretrain weight of mask generation module.")
    parser.add_argument('--save_dir_name', type=str, default='CA_exp002_tst1', help="The directory name of saving sample images.")
    parser.add_argument('--test_set_name', type=str, default='tstSet', help="The name of testing data set.")
    
    ## Generating setting : 
    parser.add_argument('--generate_num', type=int, default=20001, help="The number of lower resolution with generate sr images.")
    parser.add_argument('--lr_img_dir', type=str, default='CASIAwMskLft', help="The directory name of lower resolution images (directory placed in $data_set).")
    parser.add_argument('--file_ext', type=str, default='jpg', help="The file extension of generated images.")
    
    args = parser.parse_args() # parser the arguments, get parameter via arg.parameter

    ## (2) GPU enviroment setting stage : 
    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        os.system('echo $CUDA_VISIBLE_DEVICES')
        config = tf.ConfigProto(allow_soft_placement=True)  
        config.gpu_options.allow_growth=True   # Avoid to run out all memory, allocate it depend on the requiring
        sess = tf.Session(config=config)
        KTF.set_session(sess)
        n_gpus = len(args.cuda.split(','))
        
    else:
        print('Training without gpu. It is recommended using at least one gpu.')
        if args.force_cpu:
            n_gpus = 0
        else:
            raise Exception('sorry, but without gpu without training, since the memory is not enough.')

    parser.add_argument('--data_dir', type=str, default='../dataSet/', help="The root of directory of the data set.")

    ## (3) Initialize parameter list stage : 
        ## The parameters defined in list specifically will not be recommand to modification,
        ##      otherwise you know what you're going to do.
    init_params = {
        ## Execution enviroment setting :
        'n_gpus': n_gpus,
        
        ## Image related setting :
        'lr_shape': (args.lr_height, args.lr_width, 3), # RGB
        'img_scalar': args.img_scalar,
            
        ## Model specification :
        'res_scalar':0.2,
        'n_RRDRB': args.n_RRDRB,
        'n_DRB':3,
        'n_G_filt': args.n_G_filt,
        'n_D_filt': args.n_D_filt,
        'pretrain':args.pretrain_RRDRB,
        #'D_patch_size': args.DPatSiz,  ## see HACKME.1/
        #------------------------------------------------#
        'pre_model_dir': '../pretrain/CASIA',
        
        
        ## Learning setting :
        'exe_mode': args.exe_mode,
        # see TODO.1 : Split the dictorary of high/low resolution images.
        #'tra_hr_dir': os.path.join(args.dataSet, args.traSetNam, 'HR'),
        #'tra_lr_dir': os.path.join(args.dataSet, args.traSetNam, 'LR'),
        #'tst_lr_dir': args.tstSetPath,  # see TODO.2
        #-----------------------------------------------------------------#
        'train_mode': True,
        'prepro_dir': '../preprocess_img/',
        'learn_rate': 5e-5,#1e-4,       
        'ext': '.png',               
        'loss_weights': {'pixel':1, 'att_percept':2e-2,  'gen':5e-3} ## {'pixel':1e-2, 'percept':1, 'gen':5e-3} ## 
    }   
    
    train_params={  
        'epochs' : args.epochs,
        'batch_size' : args.batch_size,
        'sample_interval' : args.samp_img_intval,
        'load_G_D_weight': args.load_G_D_weight,
        'save_generator' : args.save_generator,
        'load_msk_gen_weight': args.load_msk_gen_weight,
        'save_dir_name':args.save_dir_name,
        #-----------------------------------------#
        'data_set_name': 'CASIAwMskLft',  ## CASIA mask, CASIAwMskLft
        'train_G_ratio' : 1  ## The parameter proposed from Generator part in RGAN implementation.
    }
    
    gen_params={
        ## Generating setting : 
        'n_sr_img':args.generate_num,
        'lr_img_dir':args.lr_img_dir,
        'gen_batch_size':1,  ## batch size of generating sr imgs, too big may cause OOM (Out Of Memory) error .
        'file_ext':args.file_ext
    }
    
    ## (4) Assignments of model executing stage : 
    deepMod = ESRGAN(**init_params)
    deepMod.training(**train_params)
    #deepMod.lookback_hist(100, 'exp002/0')
    #deepMod.generating_img(**gen_params)  
    

if __name__ == '__main__':
    main()