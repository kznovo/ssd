'''utility functions/classes for SSD'''
import numpy as np
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from imageio import imread
from random import shuffle


def preprocess_data(img, gtlabels):
    '''---------------------------------------------------------------------------
    Preprocess image and label data for training.
    # input: an image and a list of ground truth labels 
                ([one_hot_vector, xmin, ymin, xmax, ymax] format)
    # output: same shape but preprocessed data
    
    # Does:
        1. Random cropping
        2. Random flipping
        3. Random color distortions
    ---------------------------------------------------------------------------'''
    if np.random.random() < 0.5:
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        target_area = np.random.random() * img_area
        random_ratio = np.random.random() * 0.25 + 0.75
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        # create new target
        new_targets = []
        for box in gtlabels:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
            new_targets.append(box)
        gtlabels = np.asarray(new_targets).reshape(-1, gtlabels.shape[1])

    if np.random.random() < 0.5:
        img = img[:, ::-1]
        gtlabels[:, [0, 2]] = 1 - gtlabels[:, [2, 0]]
    if np.random.random() < 0.5:
        img = img[::-1]
        gtlabels[:, [1, 3]] = 1 - gtlabels[:, [3, 1]]
    
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.image.random_brightness(img, max_delta=32. / 255.)
    with tf.Session() as sess:
        img = sess.run(img)
    
    return img, gtlabels



def convert_coordinates(boxes, conversion):
    '''---------------------------------------------------------------------------
    convert coordinates of boxes.
        # input:
            boxes array([[cx, cy, w, h], ...]) or array([[xmin, ymin, xmax, ymax], ...])
        # Arguments: 
            conversion=`corners2centroids` or `centroids2corners`
        # output:
            converted boxes
    ---------------------------------------------------------------------------'''
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



def iou(boxes1, boxes2):
    '''---------------------------------------------------------------------------
    Compute Intersection over Union.
    ---------------------------------------------------------------------------'''
    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)
    boxes1 = convert_coordinates(boxes1, conversion='centroids2corners')
    boxes2 = convert_coordinates(boxes2, conversion='centroids2corners')
    intersection = np.maximum(0, np.minimum(boxes1[:,2], boxes2[:,2]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,1], boxes2[:,1]))
    union = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1]) + (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1]) - intersection
    return intersection / union




class SSDUtils(object):
    '''---------------------------------------------------------------------------
    SSD utils class.
    # Encode data for training: Typically used within the generator class.
                                Input-data is delta-encoded coordinates based on default_box 
                                coordinates relative to the ground-truth bounding box coordinates.
                                defaultbox function calculates default box coordinates and match them
                                to the ground-truth coordinates,
    # Decode data for inference
    # Generator function
    1. Initiate with feature map configurations.
    2. To encode, input img file path and coordinate data
    ---------------------------------------------------------------------------'''
    def __init__(self, model, fmap,
                 input_shape=(300,300,3),
                 scale=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                 variances=[0.1, 0.1, 0.2, 0.2],
                 batch_size=None,
                 img_folderpath=None,
                 gtlabels=None,
                 trainvalratio=0.8):
        self.model = model
        self.fmap = fmap
        self.input_shape = input_shape
        self.img_size = (input_shape[1], input_shape[0])
        self.img_width = input_shape[1]
        self.img_height = input_shape[0]
        self.scale = np.array(scale) * input_shape[1]
        self.variances = variances
        self.batch_size = batch_size
        self.img_folderpath = img_folderpath
        self.gtlabels = gtlabels
        self.trainvalratio = trainvalratio
        
        if gtlabels:
            keys = sorted(gtlabels.keys())
            num_train = int(round(trainvalratio * len(keys)))
            self.train_keys = keys[:num_train]
            self.train_batches = len(self.train_keys)
            self.val_keys = keys[num_train:]
            self.val_batches = len(self.val_keys)
        
        # make default boxes
        defboxes = []
        for i,name in enumerate(fmap):
            fmap_height = [x.output_shape[1] for i,x in enumerate(model.layers) if x.name == name][0]
            fmap_width = [x.output_shape[2] for i,x in enumerate(model.layers) if x.name == name][0]
            fmap[name].append(1.0)
            num_boxes_perlocation = len(fmap[name])
            defboxes_tensor = np.zeros((fmap_height, fmap_width, num_boxes_perlocation, 4)) # 4 for coordinates
            
            # calculate width and height of each feature layer according to the aspect ratio.
            defbox_wh = []
            for ar in fmap[name]:
                if ar == 1:
                    if len(defbox_wh) == 0:
                        defbox_height = defbox_width = self.scale[i]
                        defbox_wh.append((defbox_width, defbox_height))
                    else:
                        defbox_height = defbox_width = np.sqrt(self.scale[i] * self.scale[i+1])
                        defbox_wh.append((defbox_width, defbox_height))
                else:
                    defbox_width = self.scale[i] * np.sqrt(ar)
                    defbox_height = self.scale[i] / np.sqrt(ar)
                    defbox_wh.append((defbox_width, defbox_height))
            defbox_wh = np.array(defbox_wh)
            
            #calculate center xy coordinates of default boxes
            step_x = self.img_width / fmap_width
            step_y = self.img_height / fmap_height
            line_x = np.linspace(0.5 * step_x, self.img_width - (0.5 * step_x), fmap_width)
            line_y = np.linspace(0.5 * step_y, self.img_height - (0.5 * step_y), fmap_height)
            defbox_cx, defbox_cy = np.meshgrid(line_x, line_y)
            
            # concat together center xy and width and height of default boxes 
            defbox_cx = np.expand_dims(defbox_cx, -1)
            defbox_cy = np.expand_dims(defbox_cy, -1)
            defboxes_tensor[:, :, :, 0] = np.tile(defbox_cx, (1, 1, num_boxes_perlocation))
            defboxes_tensor[:, :, :, 1] = np.tile(defbox_cy, (1, 1, num_boxes_perlocation))
            defboxes_tensor[:, :, :, 2] = defbox_wh[:, 0]
            defboxes_tensor[:, :, :, 3] = defbox_wh[:, 1]
            
            # normalize
            defboxes_tensor[:, :, :, 0::2] /= self.img_width
            defboxes_tensor[:, :, :, 1::2] /= self.img_height
            
            # reshape and append to list
            defboxes_tensor = np.expand_dims(defboxes_tensor, axis=0)
            defboxes_tensor = defboxes_tensor.reshape(-1,4)
            defboxes.append(defboxes_tensor)
        
        # concatenate the list to a single tensor
        defboxes = np.concatenate(defboxes)
        self.defboxes = defboxes
            
    
    def _encode(self, gtlabels):
        '''-----------------------------------------------------------------------
        # input: ground truth label [one_hot_vector w/ background, xmin, ymin, xmax, ymax]
                and default box labels [cx, cy, w, h]
        # output: deltaencoded labels. only for iou threshold of above 0.5.
                shape -> ( #gtboxes, #defboxes, #class + 4 )
                For those that had iou of below 0.5 would be labeled as background.
        
        <Steps>
        1. Compute iou of the ground truth box and default boxes
        2. Get default boxes with iou higher than 0.5
        3. Compute delta between ground truth box and default boxes based on default boxes.
        4. Make new tensor out of the computed delta
        5. Append 1 to background class and combine the class vector to tensor
        6. Compare iou and make tensor with all the best iou
        -----------------------------------------------------------------------'''
        num_gtboxes = len(gtlabels)
        
        iou_list = []
        for i in range(num_gtboxes):
            iou_list.append(iou(gtlabels[i,-4:],self.defboxes))
        iou_list = np.vstack(iou_list)
        iou_mask = iou_list > 0.5
        
        encoded = np.zeros((gtlabels.shape[0], self.defboxes.shape[0], gtlabels.shape[1]))
        for i in range(num_gtboxes):
            # pick default boxes
            matched_defboxes = self.defboxes[iou_mask[i]]
            # normalize 
            encoded[i,iou_mask[i],-4::2] /= self.img_width
            encoded[i,iou_mask[i],-3::2] /= self.img_height
            # delta encode
            encoded[i,iou_mask[i],-4] = gtlabels[i,-4] - matched_defboxes[:,0]
            encoded[i,iou_mask[i],-3] = gtlabels[i,-3] - matched_defboxes[:,1]
            encoded[i,iou_mask[i],-2] = gtlabels[i,-2] - matched_defboxes[:,2]
            encoded[i,iou_mask[i],-1] = gtlabels[i,-1] - matched_defboxes[:,3]
            # encode variance
            encoded[i,iou_mask[i],-4] /= self.variances[0]
            encoded[i,iou_mask[i],-3] /= self.variances[1]
            encoded[i,iou_mask[i],-2] /= self.variances[2]
            encoded[i,iou_mask[i],-1] /= self.variances[3]
            # add category
            encoded[i,iou_mask[i],:-4] = gtlabels[i,:-4]
            # add background category
            encoded[i,np.logical_not(iou_mask[i]),0] = 1
        
        dimension_mask = np.argsort(iou_mask,axis=0) > 0
        encoded = encoded[dimension_mask]
        
        return encoded
    
    
    def generate(self, train=True):
        '''-----------------------------------------------------------------------
        Generator function for training.
        -----------------------------------------------------------------------'''
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                img_path = self.img_folderpath + key
                img = imread(img_path).astype('float32')
                gtlabels = self.gtlabels[key].copy()
                img = imresize(img, self.img_size).astype('float32')
                # if training, preprocess images and gtlabels
                if train:
                    img, gtlabels = preprocess_data(img=img, gtlabels=gtlabels)
                # encode gtlabels
                gtlabels = self._encode(gtlabels)
                # create list of inputs and targets
                inputs.append(img)
                targets.append(gtlabels)
                # append until specified batch size
                if len(targets) == self.batch_size:
                    tmp_inputs = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inputs), tmp_targets
    
    
    
    def decode(self, y_pred, confthresh=0.6):
        '''-----------------------------------------------------------------------
        Decoder function for inference
        Note that predictor could handle more than 1 input.
        y_pred would be a np.array of shape (#batch, #defboxes, #class+4)
        1. Decode and get absolute coordinates
            a. get matched default box
            b. reengineer the encoding
        2. Apply non-maximum-suppression
        -----------------------------------------------------------------------'''
        num_inputs = len(y_pred)
        # decoding
        print('decoding...')
        for i in range(num_inputs):
            # undo variances
            y_pred[i,:,-4] *= self.variances[0]
            y_pred[i,:,-3] *= self.variances[1]
            y_pred[i,:,-2] *= self.variances[2]
            y_pred[i,:,-1] *= self.variances[3]
            # undo delta-encode
            y_pred[i,:,-4] += self.defboxes[:,0]
            y_pred[i,:,-3] += self.defboxes[:,1]
            y_pred[i,:,-2] += self.defboxes[:,2]
            y_pred[i,:,-1] += self.defboxes[:,3]
        # greedy non-maximum-suppression
        print('non-maximum-suppression...')
        y_pred_decoded_nms = []
        for batch_item in y_pred: # For the labels of each batch item...
            boxes_left = np.copy(batch_item)
            maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
            while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
                maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
                maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
                maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
                boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
                if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
                similarities = iou(maximum_box[-4:], boxes_left[:,-4:]) # ...compare (IoU) the other left over boxes to the maximum box...
                boxes_left = boxes_left[similarities <= 0.45] # ...so that we can remove the ones that overlap too much with the maximum box
            
            # conf-thresh
            # 1. get all with max above conf thresh
            # 2. remove background class
            # 3. return as [category label, conf, cx, cy, w, h]
            print('conf-thresh')
            above_conf = np.array(maxima)[np.any(np.array(maxima)[:,:-4] > confthresh, axis=1)]
            category_position = above_conf[:,:-4].argmax(axis=1)
            category_conf_max = above_conf[:,:-4].max(axis=1)
            decoded = np.zeros((len(above_conf), 6))
            decoded[:,0] = category_position - 1
            decoded[:,1] = category_conf_max
            decoded[:,2:] = above_conf[:,-4:]
            decoded = decoded[decoded[:,0] >= 0]
            y_pred_decoded_nms.append(decoded)
            
        return y_pred_decoded_nms
        
        
        
        
def sample_tensors(weights_list, sampling_instructions, axes=None, init=None, mean=0.0, stddev=0.005):
    '''utility function for subsampling weights from pre-trained weights.
    '''
    first_tensor = weights_list[0]

    if (not isinstance(sampling_instructions, (list, tuple))) or (len(sampling_instructions) != first_tensor.ndim):
        raise ValueError("The sampling instructions must be a list whose length is the number of dimensions of the first tensor in `weights_list`.")

    if (not init is None) and len(init) != len(weights_list):
        raise ValueError("`init` must either be `None` or a list of strings that has the same length as `weights_list`.")

    up_sample = [] # Store the dimensions along which we need to up-sample.
    out_shape = [] # Store the shape of the output tensor here.
    # Store two stages of the new (sub-sampled and/or up-sampled) weights tensors in the following two lists.
    subsampled_weights_list = [] # Tensors after sub-sampling, but before up-sampling (if any).
    upsampled_weights_list = [] # Sub-sampled tensors after up-sampling (if any), i.e. final output tensors.

    # Create the slicing arrays from the sampling instructions.
    sampling_slices = []
    for i, sampling_inst in enumerate(sampling_instructions):
        if isinstance(sampling_inst, (list, tuple)):
            amax = np.amax(np.array(sampling_inst))
            if amax >= first_tensor.shape[i]:
                raise ValueError("The sample instructions for dimension {} contain index {}, which is greater than the length of that dimension.".format(i, amax))
            sampling_slices.append(np.array(sampling_inst))
            out_shape.append(len(sampling_inst))
        elif isinstance(sampling_inst, int):
            out_shape.append(sampling_inst)
            if sampling_inst == first_tensor.shape[i]:
                # Nothing to sample here, we're keeping the original number of elements along this axis.
                sampling_slice = np.arange(sampling_inst)
                sampling_slices.append(sampling_slice)
            elif sampling_inst < first_tensor.shape[i]:
                # We want to SUB-sample this dimension. Randomly pick `sample_inst` many elements from it.
                sampling_slice = sorted(np.random.choice(first_tensor.shape[i], sampling_inst, replace=False))
                sampling_slices.append(sampling_slice)
            else:
                # We want to UP-sample. Pick all elements from this dimension.
                sampling_slice = np.arange(first_tensor.shape[i])
                sampling_slices.append(sampling_slice)
                up_sample.append(i)
        else:
            raise ValueError("Each element of the sampling instructions must be either an integer or a list/tuple of integers, but received `{}`".format(type(sampling_inst)))

    # Process the first tensor.
    subsampled_first_tensor = np.copy(first_tensor[np.ix_(*sampling_slices)])
    subsampled_weights_list.append(subsampled_first_tensor)

    # Process the other tensors.
    if len(weights_list) > 1:
        for j in range(1, len(weights_list)):
            this_sampling_slices = [sampling_slices[i] for i in axes[j-1]] # Get the sampling slices for this tensor.
            subsampled_weights_list.append(np.copy(weights_list[j][np.ix_(*this_sampling_slices)]))

    if up_sample:
        # Take care of the dimensions that are to be up-sampled.

        out_shape = np.array(out_shape)

        # Process the first tensor.
        if init is None or init[0] == 'gaussian':
            upsampled_first_tensor = np.random.normal(loc=mean, scale=stddev, size=out_shape)
        elif init[0] == 'zeros':
            upsampled_first_tensor = np.zeros(out_shape)
        else:
            raise ValueError("Valid initializations are 'gaussian' and 'zeros', but received '{}'.".format(init[0]))
        # Pick the indices of the elements in `upsampled_first_tensor` that should be occupied by `subsampled_first_tensor`.
        up_sample_slices = [np.arange(k) for k in subsampled_first_tensor.shape]
        for i in up_sample:
            # Randomly select across which indices of this dimension to scatter the elements of `new_weights_tensor` in this dimension.
            up_sample_slices[i] = sorted(np.random.choice(upsampled_first_tensor.shape[i], subsampled_first_tensor.shape[i], replace=False))
        upsampled_first_tensor[np.ix_(*up_sample_slices)] = subsampled_first_tensor
        upsampled_weights_list.append(upsampled_first_tensor)

        # Process the other tensors
        if len(weights_list) > 1:
            for j in range(1, len(weights_list)):
                if init is None or init[j] == 'gaussian':
                    upsampled_tensor = np.random.normal(loc=mean, scale=stddev, size=out_shape[axes[j-1]])
                elif init[j] == 'zeros':
                    upsampled_tensor = np.zeros(out_shape[axes[j-1]])
                else:
                    raise ValueError("Valid initializations are 'gaussian' and 'zeros', but received '{}'.".format(init[j]))
                this_up_sample_slices = [up_sample_slices[i] for i in axes[j-1]] # Get the up-sampling slices for this tensor.
                upsampled_tensor[np.ix_(*this_up_sample_slices)] = subsampled_weights_list[j]
                upsampled_weights_list.append(upsampled_tensor)

        return upsampled_weights_list
    else:
        return subsampled_weights_list