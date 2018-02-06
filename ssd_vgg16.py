from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Concatenate, Activation
from keras.regularizers import l2
from keras.models import Model
import math

from layers import L2Normalize, DefaultBox

"""SSD model based on VGG16"""
def SSD(input_shape,
        num_classes,
        fmap = {'conv4_3_norm': [1.0, 2.0, 1.0/2.0],
                'fc7': [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],
                'conv6_2': [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],
                'conv7_2': [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],
                'conv8_2': [1.0, 2.0, 1.0/2.0],
                'conv9_2': [1.0, 2.0, 1.0/2.0]}):
    
    img_size = (input_shape[1], input_shape[0])
    weight_decay=0.0005
    same_args = { 'padding':'same', 'activation':'relu', 'kernel_regularizer':l2(weight_decay), }
    valid_args = { 'padding':'valid', 'activation':'relu', 'kernel_regularizer':l2(weight_decay), }
    pool_args = { 'padding':'same', }
    score_args = { 'padding':'same', 'kernel_regularizer':l2(weight_decay), }
    
    T = dict()
    T['data'] = Input(shape=input_shape, name='data')
    T['conv1_1'] = Conv2D(64, 3, name='conv1_1', **same_args)(T['data'])
    T['conv1_2'] = Conv2D(64, 3, name='conv1_2', **same_args)(T['conv1_1'])
    T['pool1'] = MaxPooling2D(2, 2, name='pool1', **pool_args)(T['conv1_2'])
    T['conv2_1'] = Conv2D(128, 3, name='conv2_1', **same_args)(T['pool1'])
    T['conv2_2'] = Conv2D(128, 3, name='conv2_2', **same_args)(T['conv2_1'])
    T['pool2'] = MaxPooling2D(2, 2, name='pool2', **pool_args)(T['conv2_2'])
    T['conv3_1'] = Conv2D(256, 3, name='conv3_1', **same_args)(T['pool2'])
    T['conv3_2'] = Conv2D(256, 3, name='conv3_2', **same_args)(T['conv3_1'])
    T['conv3_3'] = Conv2D(256, 3, name='conv3_3', **same_args)(T['conv3_2'])
    T['pool3'] = MaxPooling2D(2, 2, name='pool3', **pool_args)(T['conv3_3'])
    T['conv4_1'] = Conv2D(512, 3, name='conv4_1', **same_args)(T['pool3'])
    T['conv4_2'] = Conv2D(512, 3, name='conv4_2', **same_args)(T['conv4_1'])
    T['conv4_3'] = Conv2D(512, 3, name='conv4_3', **same_args)(T['conv4_2'])
    T['conv4_3_norm'] = L2Normalize(20, name='conv4_3_norm')(T['conv4_2']) #conv4_3_norm 38x38
    T['pool4'] = MaxPooling2D(2, 2, name='pool4', **pool_args)(T['conv4_3'])
    T['conv5_1'] = Conv2D(512, 3, name='conv5_1', **same_args)(T['pool4'])
    T['conv5_2'] = Conv2D(512, 3, name='conv5_2', **same_args)(T['conv5_1'])
    T['conv5_3'] = Conv2D(512, 3, name='conv5_3', **same_args)(T['conv5_2'])
    T['pool5'] = MaxPooling2D(3, 1, name='pool5', **pool_args)(T['conv5_3'])
    T['fc6'] = Conv2D(1024, 3, name='fc6', dilation_rate=(6, 6), **same_args)(T['pool5'])
    T['fc7'] = Conv2D(1024, 1, name='fc7', **same_args)(T['fc6']) #fc7 19x19
    T['conv6_1'] = Conv2D(256, 1, name='conv6_1', **same_args)(T['fc7'])
    T['conv6_2'] = Conv2D(512, 3, name='conv6_2', strides=2, **same_args)(T['conv6_1']) #conv6_2 10x10
    T['conv7_1'] = Conv2D(128, 1, name='conv7_1', **same_args)(T['conv6_2'])
    T['conv7_2'] = Conv2D(256, 3, name='conv7_2', strides=2, **same_args)(T['conv7_1']) #conv7_2 5x5
    T['conv8_1'] = Conv2D(128, 1, name='conv8_1', **same_args)(T['conv7_2'])
    T['conv8_2'] = Conv2D(256, 3, name='conv8_2', strides=1, **valid_args)(T['conv8_1']) #conv8_2 3x3
    T['conv9_1'] = Conv2D(128, 1, name='conv9_1', **same_args)(T['conv8_2'])
    T['conv9_2'] = Conv2D(256, 3, name='conv9_2', strides=1, **valid_args)(T['conv9_1']) #conv9_2 1x1
    
    for i,x in enumerate(fmap):
        loc = x + '_mbox_loc'
        conf = x + '_mbox_conf'
        T[loc] = Conv2D((len(fmap[x]) + 1) * 4, 3, name=loc, **score_args)(T[x])
        T[conf] = Conv2D((len(fmap[x]) + 1) * num_classes, 3, name=conf, **score_args)(T[x])
        T[loc+'_reshape'] = Reshape((-1, 4), name=loc+'_reshape')(T[loc])
        T[conf+'_reshape'] = Reshape((-1, num_classes), name=conf+'_reshape')(T[conf])
        
    T['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([T[x + '_mbox_loc_reshape'] for x in fmap])
    T['mbox_conf'] = Concatenate(axis=1)([T[x + '_mbox_conf_reshape'] for x in fmap])
    T['mbox_conf'] = Activation('softmax', name='mbox_conf')(T['mbox_conf'])
    T['predictions_ssd'] = Concatenate(axis=2, name='predictions_ssd')([T['mbox_loc'], T['mbox_conf']])
    model = Model(T['data'],T['predictions_ssd'])
    return model