'''keras package : the high-level module for deep learning '''
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Add, Flatten, GlobalAveragePooling2D, LeakyReLU
from keras.layers.convolutional import Conv2D

def build_discriminator(hr_shape, num_of_filts, plus_selu=True):
    def d_block(layer_input, filters, strides=1, bn=True, kerSiz=3):
        if plus_selu:
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=kerSiz, strides=strides, padding='same',\
                       activation='selu', kernel_initializer='lecun_uniform')(layer_input)
            return d
        else:
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=kerSiz, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
    
    
    in_tensor = Input(shape=hr_shape)
    d1 = Conv2D(num_of_filts, strides=1, kernel_size=3, padding='same',\
                activation='selu', kernel_initializer='lecun_uniform')(in_tensor)
    d_rec = Conv2D(num_of_filts, strides=2, kernel_size=4, padding='same',\
                activation='selu', kernel_initializer='lecun_uniform')(d1)

    for idx in range(0, 4):
        mul_filt = 2**idx
        d_rec = d_block(layer_input=d_rec, filters=num_of_filts*mul_filt)
        d_rec = d_block(layer_input=d_rec, filters=num_of_filts*mul_filt, strides=2, kerSiz=4)

    d8 = d_rec
    d9 = GlobalAveragePooling2D()(d8)
    d10 = Dense(100, activation='selu', kernel_initializer='lecun_uniform')(d9)
    validity = Dense(1, kernel_initializer='glorot_uniform')(d10)  ## For sure the range of relativistic average loss, activation='sigmoid'
    return Model(inputs=in_tensor, outputs=validity, name='Discriminator')
    