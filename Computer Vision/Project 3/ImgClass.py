import numpy as np
import cv2
import functions as f
from copy import copy
from skimage.util.shape import view_as_windows

class ImageModel():
    def __init__(self):
        pass

    def __init__(self, imgPath: str):
        self.imgPath = imgPath

        self.imgByte = cv2.imread(self.imgPath , -1)

        if(len(self.imgByte.shape) == 2):
            self.imgType = 'grayscale'
            self.grayscale = copy(self.imgByte)
            self.imgFlat = self.imgByte.flatten()

        else:
            self.imgType = 'RGB'
            self.grayscale = cv2.cvtColor(self.imgByte, cv2.COLOR_BGR2GRAY)
            # (self.b, self.g, self.r) = self.imgByte[:, :, 0], self.imgByte[:, :, 1], self.imgByte[:, :, 2]
        

                

        self.height = self.imgByte.shape[0]
        self.width = self.imgByte.shape[1]
       

        

    
    def Harris_SIFT(self , image2_gray, img2 , matcher):
        if (matcher == 'SSD'):
           
            output = f.Harris_SIFT(self.grayscale,image2_gray,self.imgByte ,img2,'SSD')

        else:
           
           output = f.Harris_SIFT(self.grayscale,image2_gray,self.imgByte, img2,'normalized cross correlation')

        return output

    