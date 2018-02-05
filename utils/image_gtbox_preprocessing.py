'''Preprocessing image for training.'''
import cv2

class PreprocessImagesBBoxes(object):
    def __init__(self,
                 img_folderpath,
                 gtbox_coords):
        
        self.img_folderpath = img_folderpath
        self.gtbox_coords = gtbox_coords
        self.keys = gtbox_coords.keys()
        
    def _random_crop(self, image, gtboxes):
        
        return image, gtboxes
    
        
    def _random_flip(self, image, gtboxes):
        
        return image, gtboxes
    
        
    def _random_color(self, image, gtboxes):
        
        return image, gtboxes
    
        
    def preprocess(self, image, gtboxes):
        
        return image, gtboxes
        