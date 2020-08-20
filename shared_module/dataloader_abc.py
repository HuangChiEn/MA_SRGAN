#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:26:55 2020

@author: from github
"""
import scipy
import os
import re
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf
import scipy.io as sio
from os.path import join
from keras.utils import np_utils
    
"""
## Image Augumentation :
from keras.util.preprocess import ImageGenerator
dataGen = ImageGenerator(..., target_size=resize_shape) 
"""

class DataLoader():    
    def __init__(self, data_set_name=None, prepro_dir=None, hr_img_size=(128, 128), scalr=4):
        assert data_set_name is not None
        self.data_set_name = data_set_name
        self.prepro_dir = prepro_dir   ## no preprocessing now
        self.hr_img_size = hr_img_size ## still using resize down sampling
        self.msk_shape = self.hr_img_size
        self.info_dict_buffer = {'msk_tag':False, 'batch_imgs_path':[], 'rand_flip':[]}
        self.scalr = scalr

    def load_correspond_mask(self):
        
        def converttostr(input_seq, seperator):
           # Join all the strings in list
           final_str = seperator.join(input_seq)
           return final_str
       
        if self.info_dict_buffer['msk_tag'] == False:
            raise Exception("The corresponding data has not be loaded, you should \
                            call load_data function first")
        imgs_msk = []  
        rand_filp = self.info_dict_buffer['rand_flip']
        for idx, img_path in enumerate(self.info_dict_buffer['batch_imgs_path']):
            path_lst = img_path.split('/')
            file_name_lst = path_lst.pop(-1).split('.')
            file_name = file_name_lst[-2] + "_mask." + file_name_lst[-1] 
            msk_dir = converttostr(path_lst, '/')
            img_msk_path = msk_dir + "/msk/" + file_name
            img_msk = imageio.imread(img_msk_path)
            img_msk = scipy.misc.imresize(img_msk, self.hr_img_size)
            if rand_filp and rand_filp[idx] < 0.5:
                img_msk = np.fliplr(img_msk)
            
            imgs_msk.append(img_msk)
            
        imgs_msk = np.array(imgs_msk)
        
        ## resetting the buffer.
        self.info_dict_buffer = {'msk_tag':False, 'batch_imgs_path':[], 'rand_flip':[]}
        return imgs_msk
    
    '''The Generator based no duplicate image'''
    def ld_data_gen(self, batch_size=1, fliplr=False, include_msk=False, shuffled=True):
        ## glob file name with reg_exp.
        def glob_reg_exp(exp=r"/*", invert=False):
            re_template = re.compile(exp)
            serch_dir = ('../datasets/%s'%(self.data_set_name))
            
            if invert is False:
                file_names = [ x for x in os.listdir( serch_dir) if re_template.search(x)]
            else:
                file_names = [ x for x in os.listdir( serch_dir ) if not re_template.search(x)]
                
            map_gen = map(lambda x: os.path.join(serch_dir, x), file_names)
            file_paths = [ path for path in map_gen]
            return file_paths
            
        ## generator of loading training image.
        def get_batch_img(imgs_path):
            iteration = len(imgs_path)//batch_size
            itr_obj = iter(imgs_path)
            
            [h, w] = self.hr_img_size
            low_h, low_w = int(h / self.scalr), int(w / self.scalr)
            
            for _ in range(iteration):  ## load one img per next()
                filNam, imgs, hr_imgs, lr_imgs = [], [], [], []
                batch_imgs_path, rand_num = [], []  ## for loading msk with random style.
                
                for idx in range(batch_size):
                    path = next(itr_obj)
                    batch_imgs_path.append(path)
                    img = self.__imread(path)
                    file_name = path.split('/')[-1].split('.')[-2]  # get file name 
                    
                    img_hr = scipy.misc.imresize(img, self.hr_img_size)
                    img_lr = scipy.misc.imresize(img_hr, (low_h, low_w))
                
                    # If training => do random flip
                    rand_num.append(np.random.random())
                    if fliplr and rand_num[idx] < 0.5:
                        img, img_hr, img_lr = np.fliplr(img), np.fliplr(img_hr), np.fliplr(img_lr)
        
                    filNam.append(file_name)
                    imgs.append(img)
                    hr_imgs.append(img_hr)
                    lr_imgs.append(img_lr)
                    
                ## For asynchronous loading :
                if include_msk:
                    self.info_dict_buffer['msk_tag'] = include_msk
                    self.info_dict_buffer['batch_imgs_path'] = batch_imgs_path
                    self.info_dict_buffer['rand_flip'] = rand_num
                    
                imgs = np.array(imgs) / 127.5 - 1.
                hr_imgs = np.array(hr_imgs) / 127.5 - 1.
                lr_imgs = np.array(lr_imgs) / 127.5 - 1.
          
                yield filNam, imgs, hr_imgs, lr_imgs
        
        ## Setting the limitation of glob range.. (for each class glob 10 image)
        all_path = glob_reg_exp('/*.jpg')
        # tra_path = glob_reg_exp('/cls[0-9]*_[0-6].jpg')
        # val_path = glob_reg_exp('/cls[0-9]*_[7-9].jpg')
        
        shuffled and np.random.shuffle(all_path) ## already randomization the path
        rnd_path = all_path
        
        ld_datagen = get_batch_img(rnd_path)
        
        return ld_datagen
      

    def ld_data_rnd(self, batch_size=1, fliplr=False, include_msk=False):
        
        path = glob('../datasets/%s/*.jpg' % (self.data_set_name))
        
        batch_imgs_path = np.random.choice(path, size=batch_size)

        imgs = []
        imgs_hr = []
        imgs_lr = []
        rand_num = []
        
        for idx, img_path in enumerate(batch_imgs_path):
            img = self.__imread(img_path)
            
            h, w = self.hr_img_size
            low_h, low_w = int(h / self.scalr), int(w / self.scalr)

            img_hr = scipy.misc.imresize(img, self.hr_img_size)
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            rand_num.append(np.random.random())
            if fliplr and rand_num[idx] < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)
            imgs.append(img)
            
        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        imgs = np.array(imgs) / 127.5 - 1.
        
        ## For asynchronous loading :
        if include_msk:
            self.info_dict_buffer['msk_tag'] = include_msk
            self.info_dict_buffer['batch_imgs_path'] = batch_imgs_path
            if fliplr:
                self.info_dict_buffer['rand_flip'] = rand_num
        
        return imgs_hr, imgs_lr, imgs
    
    def get_img_generator(self, batch_size=1, given_dataset_name=None, file_ext=None):
        
        ## generator of loading all image in dataset.
        def __get_batch(imgs_path):
            h, w = self.hr_img_size
            low_h, low_w = int(h / self.scalr), int(w / self.scalr)
            
            for path in imgs_path:  ## load one img per next()
                lst = []
                ## Sorry, but for now we only have high resolution
                img_hr = self.__imread(path)
                img_lr = scipy.misc.imresize(img_hr, (low_h, low_w))
                lst.append(img_lr)
                bt_img_lr = np.array(lst) / 127.5 -1
                file_name = path.split('/')[-1].split('.')[-2]  # get file name 
                yield bt_img_lr, file_name
                
        assert(file_ext is not None), "ERROR_MESSAGE : The given file extension is None.."
        imgs_path = glob('../datasets/%s/*.%s' % (self.data_set_name, file_ext)) \
            if given_dataset_name is None \
                else glob('../datasets/%s/*.%s' % (given_dataset_name, file_ext))
        self.batch_size = batch_size
        imgs_generator = __get_batch(imgs_path)
        return imgs_generator

    def load_imgs(self, path, all_img=True):
        imgs_path = glob(path+'*.jpg')
        return imageio.imread(imgs_path, pilmode='RGB').astype(np.float)

    def __imread(self, path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)
    
def load_dtd(dtd_dir, shuffled, batch_size):
    
    def get_lab_dict(all_path):
        lab_set = set()
        for path in all_path:
            name_lst = path[0].split('/')
            lab_set.add(name_lst[0])
        lab_set = sorted(list(lab_set))
        lab_dict = \
            dict( [(item, idx) \
                   for idx, item in enumerate(lab_set)] )
        return lab_dict
    
    def one_hot_encoding(batch_label):
        batch_oneHot = []
        for labIdx in batch_label:
            init_vec = np.zeros([47])
            init_vec[labIdx] = 1
            batch_oneHot.append(init_vec)
        return batch_oneHot
    
    def data_gen(all_path, lab_dict):
        iteration = len(all_path)//batch_size
        itr_obj = iter(all_path)
        
        for _ in range(iteration):
            batch_img, batch_lab = [], []
            for _ in range(batch_size):
                path = next(itr_obj)
                
                name_lst = path[0].split('/')
                label = lab_dict[ name_lst[0] ]
                batch_lab.append(label)
                
                full_path = join(dtd_dir, 'images', path[0])
                
                img = imageio.imread(full_path, pilmode='RGB').astype(np.float)
                img = scipy.misc.imresize(img, (480, 640, 3))
                batch_img.append(img)
                
            batch_img = np.array(batch_img) / 127.5 - 1
            batch_lab = np.array(one_hot_encoding(batch_lab))
            yield batch_img, batch_lab
        
    mat_img_path = sio.loadmat(dtd_dir+'/imdb/imdb.mat')
    zip_tmp = mat_img_path['images'][0]
    img_path = zip_tmp[0][1][0]
    shuffled and np.random.shuffle(img_path)
    lab_dict = get_lab_dict(img_path)
    
    return data_gen(img_path, lab_dict), lab_dict

## Not recommand to use, unless the small dataset.
def load_all_imgs(batch_size=1, given_dataset_name=None):
    if given_dataset_name is None:
        img_path = glob('../datasets/%s/*' % ('tmp'))
    else:
        img_path = glob('../datasets/%s/*' % (given_dataset_name))
    
    imgs_lr = []
    for path in img_path:
        img_lr = imageio.imread(img_path)
        imgs_lr.append(img_lr)
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.
    
    return imgs_lr
        
## TODO : Refactor..
if __name__ == "__main__":
    data_gen, _ = load_dtd('../datasets/dtd', False, 16)
    bat_img, bat_lab = next(data_gen)
    print(bat_lab)

