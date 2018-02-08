import tensorflow as tf

class MultiboxLoss(object):
    '''---------------------------------------------------------------------------
    Compute loss for ssd.

    <Steps>
    1. positives: just sum.
    2. negatives:
        a. Compute the classification loss for all negative boxes
        b. Make negative mask using indices and values from tf.nn.top_k
        c. Identify the top-k boxes with the highest confidence loss that 
            belong to the background class in the ground truth data.
    ---------------------------------------------------------------------------'''
    def __init__(self,
                 num_classes,
                 alpha=1.0,
                 neg_pos_ratio=3.0):
        
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio

    def _smoothl1_loss(self, y_true, y_pred):
        '''location: smooth L1 loss'''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)
    
    def _softmax_loss(self, y_true, y_pred):
        '''categorization: softmax loss'''
        y_pred = tf.maximum(y_pred, 1e-15)
        softmax_loss = - tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return softmax_loss
    
    def compute_loss(self, y_true, y_pred):
        # compute each conf_loss and loc_loss
        loc_loss = tf.to_float(self._smoothl1_loss(y_true[:, :, -4:], y_pred[:, :, -4:]))
        conf_loss = tf.to_float(self._softmax_loss(y_true[:, :, :-4], y_pred[:, :, :-4]))
        
        # create mask and count for utils
        positive_mask = tf.to_float(tf.reduce_max(y_true[:, :, 1:-4], axis=-1))
        negative_mask = y_true[: ,: ,0]
        num_positives = tf.reduce_sum(positive_mask)
        num_negatives = num_positives * self.neg_pos_ratio
        
        # get location losses: only positives, because there's no ground truth location for background
        location_loss = tf.reduce_sum(loc_loss * positive_mask, axis=1)
        
        # get classification losses
        positive_conf_loss = tf.reduce_sum(conf_loss * positive_mask, axis=1)
        negative_conf_loss_all = conf_loss * negative_mask
        negative_conf_loss_all_flat = tf.reshape(negative_conf_loss_all, [-1])
        values, indices = tf.nn.top_k(negative_conf_loss_all_flat, k=num_negatives)
        negative_mask_topk = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(negative_conf_loss_all_flat))
        negative_mask_topk = tf.to_float(tf.reshape(negative_mask_topk, [batch_size, tf.shape(y_true)[1]]))
        negative_conf_loss = tf.reduce_sum(conf_loss * negative_mask_topk, axis=-1)
        classification_loss = positive_conf_loss + negative_conf_loss
        
        # combine location and classification losses
        total_loss = classification_loss + self.alpha * location_loss
        return total_loss
    