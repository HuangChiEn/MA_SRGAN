from keras.models import Model
from keras.layers import Add, Concatenate, Input, Dense
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.core import Lambda
import keras.backend as K
import tensorflow as tf 
## HACKME.0 : https://distill.pub/2016/deconv-checkerboard/
##          https://arxiv.org/ftp/arxiv/papers/1609/1609.07009.pdf
##          The subpixel convolution is one of the upsampling method, to confirm the kernel size divide by stride
##              to eliminate the checkboard effect, however, it can not totally prevent the effect..
'''
=========================================<Subpixel upsampling>======================================
def SubpixelConv2D(input_shape, scale=2):
    def subpixel_shape(input_shape):
        dims = [input_shape[0], 
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3]/(scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    return Lambda(lambda x: tf.depth_to_space(x, scale), output_shape=subpixel_shape)
#-------------------------------------------------------------------------------------------------#

@ For activate the subpixel function, from keras.layers import Activation first,
    please insert the following code into the upsample function.
        ##x = self.SubpixelConv2D(x.shape)(x);x = Activation('tanh')(x)
    
====================================================================================================
'''

'''
==========================================<Bicubic upsampling>======================================
def bicubic(tensor, height, width):
            return Lambda(lambda x : tf.image.resize_bicubic(x, size=(height*2, width*2)))(tensor)
            
#-------------------------------------------------------------------------------------------------#

@ For activate the subpixel function, from keras.layers import Activation first,
    please insert the following code into the upsample function.
        ##lst = x.get_shape().as_list()
        ##x = Lambda(lambda tensor : bicubic(tensor, height=lst[1], width=lst[2]))(x)
    
====================================================================================================
'''
## HACKME.1 : https://mlfromscratch.com/activation-functions-explained/
##          In currently, I attempt to use SELU to prevent any gradient related problem,
##              however, the speed will slower due to expotential computation.  
##          https://arxiv.org/pdf/1905.01338.pdf ; and the effect is unknown for SELU.

##          Add. https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
##          The above tips for training GAN suggest ReLU to prevent gradient vanishment,
##              I think that, we use SELU is suitable for solve this probelm.

## HACKME.2 : For some realiable resource(the paper I forgot the name), 
##              the checkboard effect can be totally eliminated in Subpixel Conv.
##              Subpixel_Conv + Zero_order_hold -> image without checkboard effect. 
##              Zero_order_hold : 'nearest interpolation' in computer vision.

def build_generator(lr_shape, num_of_filts, num_of_RRDRB, num_of_DRB, resScal, upScalar, residual_plus=True, n_residual_plus=True):
    
    def learned_gaussian_noise(input_tensor):
        def gau_noise():  ## return 1-dim tensor of noise by "expand_dims" function.
            return K.expand_dims(K.random_normal(shape=(1,), mean=0.0, stddev=1.0))
        
        scalar_noise = Dense(1, kernel_initializer='Ones', use_bias=False)(gau_noise())
        return Lambda(lambda x : x+scalar_noise)(input_tensor)
    
    def residual_dense_block(input_tensor, filters, scale=0.2):
        outer_skip_connt = inner_skip_connt = x_1 = input_tensor
        ## inner-structure : residual one Dense block 
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',\
                   activation='selu', kernel_initializer='lecun_uniform')(input_tensor)
        x = x_2 = Concatenate(axis=-1)([x_1, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', \
                   activation='selu', kernel_initializer='lecun_uniform')(x)
        x = x_3 = Concatenate(axis=-1)([x_2, x])
        
        ## Add residual connection per 2 convolution block.
        #   Down-dimension via express the filter channel with 1x1 convBlock.
        lst = inner_skip_connt.get_shape().as_list()
        x = Conv2D(filters=lst[-1], kernel_size=1)(x)
        x = inner_skip_connt = Add()([inner_skip_connt, x])  
        
        ## inner-structure : residual two Dense block 
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', \
                   activation='selu', kernel_initializer='lecun_uniform')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = x_4 = Concatenate(axis=-1)([x_3, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', \
                   activation='selu', kernel_initializer='lecun_uniform')(x)
        x = Concatenate(axis=-1)([x_4, x])     
        
        ## Add residual connection per 2 convolution block.
        lst = inner_skip_connt.get_shape().as_list()
        x = Conv2D(filters=lst[-1], kernel_size=1)(x)
        x = Add()([inner_skip_connt, x])  ## Add residual connection..
        
        ## outer-structure : conv2D + resodual connection 
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        ## Residual Scaling && skip connection in dense block : 
        x = Lambda(lambda x: x * resScal)(x)
        x = Add()([outer_skip_connt, x])
        return x
        
    ## Original dense block consist of 5 convolution block with outter residual connection.
    def dense_block(input_tensor, filters, scale=0.2):  
        skip_connt = x_1 = input_tensor
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
        x = LeakyReLU(alpha=0.2)(x)
        x = x_2 = Concatenate(axis=-1)([x_1, x])  # Note : axis=-1, append the feature map in last channel.
                                                  #  It will increase the num of fileter in G, request more memory!!
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = x_3 = Concatenate(axis=-1)([x_2, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = x_4 = Concatenate(axis=-1)([x_3, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Concatenate(axis=-1)([x_4, x])  ## see HACKME.2
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        ## Residual Scaling && skip connection in dense block : 
        x = Lambda(lambda x: x * resScal)(x)
        x = Add()([skip_connt, x])
        return x
    
    def upsample(in_tensor, filters, scalar=2): 
        
        def bicubic(tensor, height, width):
            return Lambda(lambda x : tf.image.resize_bicubic(x, size=(height*2, width*2)))(tensor)

        #x = UpSampling2D(size=(2, 2), interpolation='bilinear')(input_tensor)
        lst = in_tensor.get_shape().as_list()
        x = Lambda(lambda tensor : bicubic(tensor, height=lst[1], width=lst[2]))(in_tensor)  ## see HACKME.0
        out_tensor = Conv2D(filters, kernel_size=3, strides=1, padding='same', \
                   kernel_initializer='glorot_uniform')(x) 
         
        return out_tensor
    
    inputs = Input(shape=lr_shape)
    ## < Feature extractor structure/ > 
    out_skip = in_skip = x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same', \
                                    activation='selu', kernel_initializer='lecun_uniform')(inputs)
    
    for _ in range(num_of_RRDRB):
        ## Residual in Residual Dense Block : 
        for _ in range(num_of_DRB):
            ## Dense Blocks ( Residual / Vanilla ):
            if residual_plus:      # plus residual dense block.
                x = residual_dense_block(input_tensor=x, filters=num_of_filts)  
                if n_residual_plus:  # plus noise and residual dense block. 
                    x = learned_gaussian_noise(x)
            else:
                x = dense_block(input_tensor=x, filters=num_of_filts)
                
                
        ## out block process : (scaling and add)
        x = Lambda(lambda x: x * resScal)(x)
        x = in_skip = Add()([in_skip, x])
        
    x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same')(x)
    ## < /Feature extractor structure > 
    
    ## My HACKME  
    ## Source code of edsr do not contain residual scaling : 
    x = Lambda(lambda x: x * 0.8)(x)  ## residual scaling beta=0.8
    x = Add()([out_skip, x])
    
    for _ in range(upScalar//2):
        x = upsample(x, num_of_filts, scalar=2)
    
    x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same', \
               activation='selu', kernel_initializer='lecun_uniform')(x)
    
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)
    
    return Model(inputs=inputs, outputs=x, name='generator')
    