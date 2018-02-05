import tensorflow as tf

class MultiboxLoss(object):
    def __init__(self,
                 num_classes,
                 alpha=1.0,
                 neg_pos_ratio=3.0):
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        
    def _smoothl1_loss(self, y_true, y_pred):
        '''smooth L1 loss for localization'''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)
    
    def _softmax_loss(self, y_true, y_pred):
        '''softmax loss for categorization'''
        y_pred = tf.maximum(y_pred, 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return softmax_loss
    
    def compute_loss(self, y_true, y_pred):
        # compute conf_loss and loc_loss
        conf_loss = tf.to_float(self._softmax_loss(y_true[:,:,:-12], y_pred[:,:,:-12]))
        loc_loss = tf.to_float(self._smoothl1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8]))
        
        # 