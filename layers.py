'''Additional layer definition for SSD'''
import numpy as np
import keras.backend as K
from keras.engine.topology import InputSpec, Layer
import tensorflow as tf


class L2Normalize(Layer):
    '''Default box creation layer
        # input: (batch, height, width, channels)
        # output: same
    '''
    def __init__(self, init_gamma=20, **kwargs):
        self.init_gamma = init_gamma
        super(L2Normalize, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.init_gamma * np.ones((input_shape[3],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super(L2Normalize, self).build(input_shape)
    
    def call(self, x, mask=None):
        output = K.l2_normalize(x, 3) * self.gamma
        return output
    
    def get_config(self):
        config = {'gamma_init': self.gamma_init}
        base_config = super(L2Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
