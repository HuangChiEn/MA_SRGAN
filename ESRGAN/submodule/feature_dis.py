import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D
import keras.backend as K

## keras pretraining model : for extract activated feature
from keras.applications import VGG19  
from keras.applications.vgg19 import preprocess_input

"""
The description of feature discriminator :
    HACKME list :
        1) The feature discriminator may be replaced into other modern model
            i.e. MobileNetv3, Dense-Net..etc. but the generated image contain wrong colorful(see /images/prerpo_bug) 
"""
  
'''
## HACKME 1).  
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input



def build_test_feadis(hr_shape):
    
    img = Input(shape=hr_shape)
    prepro_img = prerpocess_test(img)
    ## alpha=1.4, adjustment
    MobNetv2 = MobileNetV2(input_tensor=prepro_img,
                                               include_top=False,
                                               weights='imagenet')
    
    img_fea = MobNetv2(prepro_img)
    ## Fine-tune layer :
    #img_fea = Conv2D(512, (3, 3),
    #             padding='same',
    #             name='block5_conv4')(img_fea)
    
    MobNetv2.trainable = False
    model = Model(inputs=img, outputs=img_fea, name='feaDis')
    
    return model



def prerpocess_test(x):
    if isinstance(x, np.ndarray):
        return preprocess_input((x + 1) * 127.5)
    else:
        return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x)

'''

def build_feature_dis(hr_shape):
    #FREEUSE.1 VGG19 with 512 feature map in last layer (fine-tune).
    #          fine-tune - loss(VGG(hr) - VGG(sr)) -> classification.
    ##---------------<VGG19 - structure : >-----------------------##
    oriVGG = VGG19(weights="imagenet", include_top=False)
    VGG_out = oriVGG.get_layer('block5_conv3').output
    VGGBef= Model(inputs=oriVGG.input, outputs=VGG_out)
    ## ----------------------------------------------------------------------##    
    
    ## prerpocess input.
    img = Input(shape=hr_shape)
    prepro_img = preprocess_vgg(img)
    img_features = VGGBef(prepro_img)
    return Model(img, img_features, name='feaDis')

def preprocess_vgg(x):
    ## Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network
    if isinstance(x, np.ndarray):
        return preprocess_input((x + 1) * 127.5)
    else:
        return Lambda(lambda x: preprocess_input(tf.add(x, 1) * 127.5))(x) 
