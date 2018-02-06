'''Additional layer definitions for SSD'''
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
               


class DefaultBox(Layer):
    '''Default box creation layer
        # input: (batch, height, width, channels)
        # output: (batch, fmap_height, fmap_width, n_boxes, 8)
        #     8 is for default box coordinates and variance values
    '''
    def __init__(self, img_size,
                 scale,
                 next_scale,
                 aspect_ratios,
                 variances,
                 **kwargs):
        self.img_size = img_size
        self.img_width = self.img_size[0]
        self.img_height = self.img_size[1]
        size = min(self.img_height, self.img_width)
        self.scale = scale * size
        self.next_scale = next_scale * size
        self.aspect_ratios = aspect_ratios
        self.variances = variances
        self.n_boxes = len(aspect_ratios) + 1
        super(DefaultBox, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        '''Arguments:
                x (tensor): 4D tensor of shape (batch, height, width, channels)
        '''
        # Define output shape
        # output: [[[[0,0,0,0,0,0,0,0](len=8) *n_boxes] *fmap_width] *fmap_height]
        batch_size, fmap_height, fmap_width, fmap_channels = x._keras_shape
        defboxes_tensor = np.zeros((fmap_height, fmap_width, self.n_boxes, 8))
        
        # get default box width and height
        # output: [[w1, h1], [w2, h2], ... [wn, hn]] (len=self.n_boxes)
        self.aspect_ratios.append(1.0)
        defbox_wh = []
        for ar in self.aspect_ratios:
            if ar == 1:
                if len(defbox_wh) == 0:
                    defbox_height = defbox_width = self.scale
                    defbox_wh.append((defbox_width, defbox_height))
                else:
                    defbox_height = defbox_width = np.sqrt(self.scale * self.next_scale)
                    defbox_wh.append((defbox_width, defbox_height))
            else:
                defbox_width = self.scale * np.sqrt(ar)
                defbox_height = self.scale / np.sqrt(ar)
                defbox_wh.append((defbox_width, defbox_height))
        defbox_wh = np.array(defbox_wh)
        
        # get default box center xy coordinates
        # output: defbox_cx = [[cx1, cx2, ...](len=fmap_width) *fmap_height]
        #         defbox_cy = same
        step_x = self.img_width / fmap_width
        step_y = self.img_height / fmap_height
        line_x = np.linspace(0.5 * step_x, (self.img_width - 0.5) * step_x, fmap_width)
        line_y = np.linspace(0.5 * step_y, (self.img_height - 0.5) * step_y, fmap_height)
        defbox_cx, defbox_cy = np.meshgrid(line_x, line_y)
        
        # add default box cxy & wh coordinates to defboxes_tensor
        # output: 
        defbox_cx = np.expand_dims(defbox_cx, -1)
        defbox_cy = np.expand_dims(defbox_cy, -1)
        defboxes_tensor[:, :, :, 0] = np.tile(defbox_cx, (1, 1, self.n_boxes))
        defboxes_tensor[:, :, :, 1] = np.tile(defbox_cy, (1, 1, self.n_boxes))
        defboxes_tensor[:, :, :, 2] = defbox_wh[:, 0]
        defboxes_tensor[:, :, :, 3] = defbox_wh[:, 1]
        
        # normalize, and add   for encoding
        # output: 
        defboxes_tensor[:, :, :, [0, 2]] /= self.img_width
        defboxes_tensor[:, :, :, [1, 3]] /= self.img_height
        defboxes_tensor[:, :, :, 4] = self.variances[0]
        defboxes_tensor[:, :, :, 5] = self.variances[1]
        defboxes_tensor[:, :, :, 6] = self.variances[2]
        defboxes_tensor[:, :, :, 7] = self.variances[3]
        
        # expand one dimension for the batch size and make output tensor
        # output:
        defboxes_tensor = np.expand_dims(defboxes_tensor, axis=0)
        defboxes_tensor = K.tile(K.constant(defboxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))
        return defboxes_tensor
    
    def compute_output_shape(self, input_shape):
        batch_size, fmap_height, fmap_width, fmap_channels = input_shape
        return (batch_size, fmap_height, fmap_width, self.n_boxes, 8)
    
    def get_config(self):
        config = {
            'img_size': self.img_size,
            'scale': self.scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'variances': list(self.variances)
        }
        base_config = super(DefaultBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
