'''utility functions for SSD'''
import numpy as np
import tensorflow as tf


def convert_coordinates(boxes, conversion):
    '''convert coordinates of boxes.
        # input:
            boxes array([[cx, cy, w, h], ...]) or array([[xmin, ymin, xmax, ymax], ...])
        # Arguments: 
            conversion=`corners2centroids` or `centroids2corners`
        # output:
            converted boxes
    '''
    converted = np.copy(boxes).astype(np.float)
    if conversion == 'corners2centroids':
        converted[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2.0 # Set cx
        converted[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2.0 # Set cy
        converted[..., 2] = boxes[..., 2] - boxes[..., 0] # Set w
        converted[..., 3] = boxes[..., 3] - boxes[..., 1] # Set h
    elif conversion == 'centroids2corners':
        converted[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0 # Set xmin
        converted[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0 # Set ymin
        converted[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0 # Set xmax
        converted[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0 # Set ymax
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'corners2centroids' and 'centroids2corners'.")
    return converted


def compute_iou(box, boxes):
    '''Compute intersection over union (jaccard overlap) between
        1 box vs n boxes.
        Used both for decoding and encoding data.
        # input: box (1 dim) & boxes (2 dim)
            coordinates in centroids: array([cx, cy ,w, h], ...)
        # output: iou for each boxes
    '''
    # Expand dims and convert coordinates for copmutation convenience
    box = np.expand_dims(box, axis=0)
    if len(boxes.shape) == 1: boxes = np.expand_dims(boxes, axis=0)
    box = convert_coordinates(box, conversion='centroids2corners')
    boxes = convert_coordinates(boxes, conversion='centroids2corners')
    
    # compute intersection
    inter_upleft = np.maximum(box[:, :2], boxes[:, :2])
    inter_botright = np.minimum(box[:, 2:4], boxes[:, 2:4])
    inter_wh = np.maximum(inter_botright - inter_upleft, 0)
    intersection = inter_wh[:, 0] * inter_wh[:, 1]
    
    # compute union
    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection
    
    # compute and return iou
    iou = intersection / union
    return iou


def decode(y_pred):
    '''decode y_pred for inference
        # input: y_pred of shape (n_batch, #class, #defboxes, 8)
        # output: y_pred_decoded of shape
            array([[class_id, confidence, xmin, ymin, xmax, ymax], ... ])
    '''
    compute_iou
    nms = tf.image.non_max_suppression(
    return





def encode():
    



    
    
    