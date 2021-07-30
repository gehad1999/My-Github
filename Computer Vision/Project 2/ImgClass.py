import numpy as np
import cv2
import functions as f
from copy import copy
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt

class ImageModel():
    def __init__(self):
        pass

    def __init__(self, imgPath: str):
        self.imgPath = imgPath

        self.imgByte = cv2.imread(self.imgPath , -1)
        self.imageSnake = plt.imread(self.imgPath)
        if(len(self.imgByte.shape) == 2):
            self.imgType = 'grayscale'
            self.imgFlat = self.imgByte.flatten()

        else:
            self.imgType = 'RGB'
            (self.b, self.g, self.r) = self.imgByte[:, :, 0], self.imgByte[:, :, 1], self.imgByte[:, :, 2]
            # (self.b, self.g, self.r) = cv2.split(self.imgByte)
            self.bflat = self.b.flatten()
            self.gflat = self.g.flatten()
            self.rflat = self.r.flatten()

                

        self.height = self.imgByte.shape[0]
        self.width = self.imgByte.shape[1]

    def to_grayScale(self):
            # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        gray_img = np.round(np.dot(self.imgByte[..., :3], [ 0.1140 , 0.5870, 0.2989]))
        # gray_img = np.round(np.dot(self.imgByte[..., :3], [0.21, 0.72, 0.07]))
        gray_img = gray_img.astype(int)
        # cv2.imwrite("gray2.png", gray_img)

        return gray_img
        
    def canny_edge_detection(self):

        if(self.imgType == 'grayscale'):
            image = copy(self.imgByte)

        else:
            image = self.to_grayScale()

        return f.canny_edge_detection(image)

    def hough_transform_lines(self):

        if(self.imgType == 'grayscale'):
            image = copy(self.imgByte)

        else:
            image = self.to_grayScale()

        return f.hough_transform_lines(image)

    def hough_transform_circles(self):

        if(self.imgType == 'grayscale'):
            image = copy(self.imgByte)

        else:
            image = self.to_grayScale()

        return f.hough_transform_circles(image)

    def activeContourFromCircle(self):

        # return f.activeContourFromCircle(self.imageSnake ,  (254, 158), 125 )
        return f.activeContourFromCircle(self.imageSnake ,  (290, 440), 125 )



    
   