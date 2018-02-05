from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Concatenate, Activation
from keras.regularizers import l2
from keras.models import Model
import math

from layers import L2Normalize, DefaultBox

"""SSD model based on VGG16"""

def SSD(input_shape,
        num_classes,
        weight_decay=0.0005,
        scale=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], # MS_COCO
        variances=[0.1, 0.1, 0.2, 0.2],
        aspect_ratios=[[1.0, 2.0, 1.0/2.0],
                       [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],
                       [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],
                       [1.0, 2.0, 3.0, 1.0/2.0, 1.0/3.0],
                       [1.0, 2.0, 1.0/2.0],
                       [1.0, 2.0, 1.0/2.0]],
        name = ['data','conv1_1','conv1_2','pool1',
                'conv2_1','conv2_2','pool2',
                'conv3_1','conv3_2','conv3_3','pool3',
                'conv4_1','conv4_2','conv4_3','pool4',
                'conv5_1','conv5_2','conv5_3','pool5',
                'fc6','fc7',
                'conv6_1','conv6_2',
                'conv7_1','conv7_2',
                'conv8_1','conv8_2',
                'conv9_1','conv9_2']):
    
    img_size = (input_shape[1], input_shape[0])
    same_args = { 'padding':'same', 'activation':'relu', 'kernel_regularizer':l2(weight_decay), }
    valid_args = { 'padding':'valid', 'activation':'relu', 'kernel_regularizer':l2(weight_decay), }
    pool_args = { 'padding':'same', }
    score_args = { 'padding':'same', 'kernel_regularizer':l2(weight_decay), }
    

    T = dict()
    T[name[0]] = Input(shape=input_shape, name=name[0])
    T[name[1]] = Conv2D(64, 3, name=name[1], **same_args)(T[name[0]])
    T[name[2]] = Conv2D(64, 3, name=name[2], **same_args)(T[name[1]])
    T[name[3]] = MaxPooling2D(2, 2, name=name[3], **pool_args)(T[name[2]])
    T[name[4]] = Conv2D(128, 3, name=name[4], **same_args)(T[name[3]])
    T[name[5]] = Conv2D(128, 3, name=name[5], **same_args)(T[name[4]])
    T[name[6]] = MaxPooling2D(2, 2, name=name[6], **pool_args)(T[name[5]])
    T[name[7]] = Conv2D(256, 3, name=name[7], **same_args)(T[name[6]])
    T[name[8]] = Conv2D(256, 3, name=name[8], **same_args)(T[name[7]])
    T[name[9]] = Conv2D(256, 3, name=name[9], **same_args)(T[name[8]])
    T[name[10]] = MaxPooling2D(2, 2, name=name[10], **pool_args)(T[name[9]])
    T[name[11]] = Conv2D(512, 3, name=name[11], **same_args)(T[name[10]])
    T[name[12]] = Conv2D(512, 3, name=name[12], **same_args)(T[name[11]])
    T[name[13]] = Conv2D(512, 3, name=name[13], **same_args)(T[name[12]])
    T[name[13]+'_norm'] = L2Normalize(20, name=name[13]+'_norm')(T[name[12]]) #conv4_3_norm 38x38
    T[name[14]] = MaxPooling2D(2, 2, name=name[14], **pool_args)(T[name[13]])
    T[name[15]] = Conv2D(512, 3, name=name[15], **same_args)(T[name[14]])
    T[name[16]] = Conv2D(512, 3, name=name[16], **same_args)(T[name[15]])
    T[name[17]] = Conv2D(512, 3, name=name[17], **same_args)(T[name[16]])
    T[name[18]] = MaxPooling2D(3, 1, name=name[18], **pool_args)(T[name[17]])
    T[name[19]] = Conv2D(1024, 3, name=name[19], dilation_rate=(6, 6), **same_args)(T[name[18]])
    T[name[20]] = Conv2D(1024, 1, name=name[20], **same_args)(T[name[19]]) #fc7 19x19
    T[name[21]] = Conv2D(256, 1, name=name[21], **same_args)(T[name[20]])
    T[name[22]] = Conv2D(512, 3, name=name[22], strides=2, **same_args)(T[name[21]]) #conv6_2 10x10
    T[name[23]] = Conv2D(128, 1, name=name[23], **same_args)(T[name[22]])
    T[name[24]] = Conv2D(256, 3, name=name[24], strides=2, **same_args)(T[name[23]]) #conv7_2 5x5
    T[name[25]] = Conv2D(128, 1, name=name[25], **same_args)(T[name[24]])
    T[name[26]] = Conv2D(256, 3, name=name[26], strides=1, **valid_args)(T[name[25]]) #conv8_2 3x3
    T[name[27]] = Conv2D(128, 1, name=name[27], **same_args)(T[name[26]])
    T[name[28]] = Conv2D(256, 3, name=name[28], strides=1, **valid_args)(T[name[27]]) #conv9_2 1x1
    
    
    fmap_list = list([name[13]+'_norm',name[20],name[22],name[24],name[26],name[28]])
    num_defaults = lambda x: len(aspect_ratios[x]) + 1
    
    for i,fmap_name in enumerate(fmap_list):
        loc = fmap_name+'_mbox_loc'
        conf = fmap_name+'_mbox_conf'
        defbox = fmap_name+'_mbox_defaultbox'
        
        T[loc] = Conv2D(num_defaults(i) * 4, 3, name=loc, **score_args)(T[fmap_name])
        T[conf] = Conv2D(num_defaults(i) * num_classes, 3, name=conf, **score_args)(T[fmap_name])
        T[defbox] = DefaultBox(img_size, scale=scale[i], next_scale=scale[i+1], aspect_ratios=aspect_ratios[i],
                               variances=variances, name=defbox)(T[fmap_name])
        T[loc+'_reshape'] = Reshape((-1, 4), name=loc+'_reshape')(T[loc])
        T[conf+'_reshape'] = Reshape((-1, num_classes), name=conf+'_reshape')(T[conf])
        T[defbox+'_reshape'] = Reshape((-1, 8), name=defbox+'_reshape')(T[defbox])
    
    T['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([T[fmap_name+'_mbox_loc_reshape'] for fmap_name in fmap_list])
    T['mbox_conf'] = Concatenate(axis=1)([T[fmap_name+'_mbox_conf_reshape'] for fmap_name in fmap_list])
    T['mbox_conf'] = Activation('softmax', name='mbox_conf')(T['mbox_conf'])
    T['mbox_defaultbox'] = Concatenate(axis=1,name='mbox_defaultbox')([T[fmap_name+'_mbox_defaultbox_reshape'] for fmap_name in fmap_list])
    
    T['predictions_ssd'] = Concatenate(axis=2, name='predictions_ssd')([T['mbox_conf'], T['mbox_loc'], T['mbox_defaultbox']])
    model = Model(T[name[0]],T['predictions_ssd'])
    
    return model