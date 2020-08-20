import os
import sys
sys.path.insert(1, '../shared_module')  
import dataloader  # Extandable for data preprocessing
from modelManager import ModMang

sys.path.insert(2, '../ESRGAN')  
from submodule import load_module

import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import keras.backend.tensorflow_backend as KTF
import numpy as np

if __name__ == "__main__":
    ## Define parameters :
    epochs, batch_size = 40, 1
    opt = Adam(lr=1e-4, amsgrad=True)
    
    ## Multi-GPU Env setting :
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  ## filter the warning.
#    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
#    os.system('echo $CUDA_VISIBLE_DEVICES')
    config = tf.ConfigProto(allow_soft_placement=True)  
    config.gpu_options.allow_growth=True   # Avoid to run out all memory, allocate it depend on the requiring
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    
    
    ## Build the model : 
    lvgg_tmp = load_module('l_vgg', {'input_shape':(480, 640, 3), 'cls_num':47})
    print('Inner Structure : ')
    lvgg_tmp.summary()
    
    lvgg = multi_gpu_model(lvgg_tmp, gpus=2)
    lvgg.compile(optimizer=opt, loss=['categorical_crossentropy'],
                 metrics=['accuracy'])
    print("Lightweight VGG")
    
    lvgg.summary()
    ## Training phase :
#    for idx in range(epochs):
#        data_gen, _ = \
#            dataloader.load_dtd('../datasets/dtd', True, batch_size)
#            
#        for bat_img, bat_lab in data_gen:
#            loss = lvgg.train_on_batch(bat_img, bat_lab)
#            print("Epochs : {} ; loss : {}".format(idx, loss))
#            
    #lvgg.load_weights('./DTD.h5')
    data_gen, _ = \
            dataloader.load_dtd('../datasets/dtd', True, batch_size)
    
    imgs, labs = [], []
    for bat_img, bat_lab in data_gen:
        imgs.append(*bat_img) ; labs.append(*bat_lab)
    
    loss, accuracy = lvgg.evaluate([imgs], [labs])
    print('Test:')
    print('Loss:', loss)
    print('Accuracy:', accuracy)
    
    train_his = lvgg.fit([imgs], [labs], epochs=200, batch_size=16, verbose=1)
    
    lvgg.save_weights('../pretrain/CASIA/fea_dis/DTD_tmp.h5')
    
    """VGG16 model for Keras.

    # Reference
    
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
        https://arxiv.org/abs/1409.1556) (ICLR 2015)
    
    """
    
    '''
    def triplet_loss(_, y_pred):
        """
        Implementation of the triplet loss function
        Arguments:
        _ -- true labels, required in Keras loss function but useless, replace into placeholder "_".
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        margin = KTF.constant(12)
        return KTF.mean(KTF.maximum(KTF.constant(0), KTF.square(y_pred[:,0,0])\
                                    - 0.5*(KTF.square(y_pred[:,1,0]) \
                                           + KTF.square(y_pred[:,2,0])) + margin))
        
    
    def triplet_loss(self, y_true, y_pred):
    
        embeddings = K.reshape(y_pred, (-1, 3, output_dim))
    
        positive_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,1]),axis=-1)
        negative_distance = K.mean(K.square(embeddings[:,0] - embeddings[:,2]),axis=-1)
        return K.mean(K.maximum(0.0, positive_distance - negative_distance + _alpha))
    
        self._model.compile(loss=triplet_loss, optimizer="sgd")
        self._model.fit(x=x,y=y,nb_epoch=1, batch_size=len(x))
    
        
    def some_distance(vects):
        x, y = vects
        return KTF.sqrt(KTF.maximum(KTF.sum(KTF.square(x - y), axis=1, keepdims=True), 
                                KTF.epsilon()))
    
    def fea_discriminator(input_shape):
        
        # Load lightwidth VGG :
        lvgg = load_module('l_vgg', 
                              {'img_shape' :  input_shape})
        
        # Define triplet framework :
        input_anchor, input_positive, input_negative = \
            Input(input_shape), Input(input_shape), Input(input_shape)
        
        output_anchor, output_positive, output_negative = \
            lvgg(input_anchor), lvgg(input_positive), lvgg(input_negative)
        
        # The Lamda layer produces output using given function. Here its Euclidean distance.
        positive_dist = \
            Lambda(some_distance, name='pos_dist')([output_anchor, output_positive])
        negative_dist = \
            Lambda(some_distance, name='neg_dist')([output_anchor, output_negative])
        tertiary_dist = \
            Lambda(some_distance, name='ter_dist')([output_positive, output_negative])
        
        return Model(inputs=[input_anchor, input_positive, input_negative], 
                         outputs=[positive_dist, negative_dist, tertiary_dist], name='lvgg')
    '''
