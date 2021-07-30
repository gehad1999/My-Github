import numpy as np
import cv2
import functions as f
from copy import copy
from skimage.util.shape import view_as_windows
from regionGrowing import Point 
import regionGrowing  
from agglomerative import AgglomerativeClustering 
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
            (self.b, self.g, self.r) = self.imgByte[:, :, 0], self.imgByte[:, :, 1], self.imgByte[:, :, 2]
        

                

        self.height = self.imgByte.shape[0]
        self.width = self.imgByte.shape[1]

    def Kmeans(self):
        
        image = cv2.cvtColor(self.imgByte , cv2.COLOR_BGR2RGB)


        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)

        segmented_image = f.kmeans(3, pixels , 10)
        segmented_image = segmented_image.reshape(image.shape)

        return segmented_image

    def Otsu(self):
        

        output = f.Otsu(self.grayscale)
        return output

    def Otsu_local(self):
        
        window = np.max([int(self.width/5), int(self.height/5)])

        output = f.Otsu_local(self.grayscale, window)
        return output

    def optimal_thresholding(self):

        output = f.optimal_thresholding(self.grayscale)
        return output
       
    def optimal_thresholding_local(self):
        window = np.max([int(self.width/6), int(self.height/6)])
        output = f.optimal_thresholding_local(self.grayscale, window)
        return output
       
    def mean_shift(self):
        

        output = f.mean_shift(self.imgByte)
        return output

    def regionGrowing(self):
        

        seeds = [Point(130,140)]
        output = regionGrowing.regionGrow(self.grayscale,seeds,5)
        return output

    def agglomerative(self):
        image = cv2.cvtColor(self.imgByte , cv2.COLOR_BGR2RGB)
        pixels = image.reshape((-1,3))
        n_clusters = 9
        agglo = AgglomerativeClustering(k=n_clusters, initial_k=25)
        agglo.fit(pixels)
        output = [[agglo.predict_center(list(pixel)) for pixel in row] for row in image]
        output = np.array(output, np.uint8)       
        
        return output
       

        

    
    

    