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
        self.noised_img = np.asarray([])

        

    
    def Histogram_Equalization(self):
        if (self.imgType == 'grayscale'):
           
            new_image = f.equalizeHist(self.imgFlat,self.height, self.width)

        else:
           
            new_B , new_G ,new_R = f.equalizeHist(self.bflat,self.height, self.width) ,f.equalizeHist(self.gflat,self.height, self.width),f.equalizeHist(self.rflat,self.height, self.width)
            new_image =  np.dstack((new_B, new_G, new_R))
        return new_image

    def normalize_img(self):

        if (self.imgType == 'grayscale'):
           
            new_image = f.normalize_img(self.imgFlat,self.height, self.width)

        else:
           
            new_B , new_G ,new_R = f.normalize_img(self.bflat,self.height, self.width) ,f.normalize_img(self.gflat,self.height, self.width),f.normalize_img(self.rflat,self.height, self.width)
            new_image =  np.dstack((new_B, new_G, new_R))
        
        return new_image

        
        
    
    def freq_lowpass_Filter(self):
        if (self.imgType == 'grayscale'):
           
            new_image = f.freq_lowpass_Filter(self.imgByte)

        else:
           
            new_B , new_G ,new_R = f.freq_lowpass_Filter(self.b) ,f.freq_lowpass_Filter(self.g),f.freq_lowpass_Filter(self.r)
            new_image =  np.dstack((new_B, new_G, new_R))
        
        return new_image
       

    def freq_highpass_Filter(self):

        if (self.imgType == 'grayscale'):
           
            new_image = f.freq_highpass_Filter(self.imgByte)

        else:
           
            new_B , new_G ,new_R = f.freq_highpass_Filter(self.b) ,f.freq_highpass_Filter(self.g),f.freq_highpass_Filter(self.r)
            
            new_image =  np.dstack((new_B, new_G, new_R))

        
        return new_image

    def to_grayScale(self):
        gray_img = np.round(np.dot(self.imgByte[..., :3], [0.21, 0.72, 0.07]))
        gray_img = gray_img.astype(int)
        # cv2.imwrite("gray2.png", gray_img)

        return gray_img


    def get_Histogram(self):

        if (self.imgType == 'grayscale'):
           
            Histogram = f.get_histogram(self.imgFlat)

        else:
           
            histo_B , histo_G , histo_R = f.get_histogram(self.bflat) ,f.get_histogram(self.gflat),f.get_histogram(self.rflat)
            Histogram = (histo_B , histo_G , histo_R)
        
        return Histogram
    
    def cumulative_crve(self):
        MN = self.height * self.width 
        if (self.imgType == 'grayscale'):
           
            out_put = f.cumulative_crve(self.imgFlat) /  MN

        else:
           
            B , G , R = f.cumulative_crve(self.bflat) ,f.cumulative_crve(self.gflat),f.cumulative_crve(self.rflat)
            out_put = (B / MN , G / MN , R / MN)
        
        return out_put

    def salt_and_pepper_noise(self):

        if (self.imgType == 'grayscale'):
           
            noise_img = f.salt_and_pepper_noise(self.imgFlat, self.height, self.width)

        else:
           
            noise_B = f.salt_and_pepper_noise(self.bflat, self.height, self.width)
            noise_G = f.salt_and_pepper_noise(self.gflat, self.height, self.width)
            noise_R = f.salt_and_pepper_noise(self.rflat, self.height, self.width)
        

            noise_img =  np.dstack((noise_B, noise_G, noise_R))
        
        return noise_img

    def median_filter(self):
        if (self.imgType == 'grayscale'):
           
            output_image = f.median_filter(self.noised_img, self.height, self.width)
          

        else:
            (nb, ng, nr) = cv2.split(self.noised_img)
            new_B = f.median_filter(nb, self.height, self.width)
            new_G = f.median_filter(ng, self.height, self.width)
            new_R = f.median_filter(nr, self.height, self.width)
           
            output_image = ( np.dstack((new_B, new_G, new_R))).astype(np.uint8)
        
        return output_image

    def average_filter(self):
        if (self.imgType == 'grayscale'):
           
            output_image = f.average_filter(self.noised_img, self.height, self.width)

        else:
            (nb, ng, nr) = cv2.split(self.noised_img)
            new_B = f.average_filter(nb, self.height, self.width)
            new_G = f.average_filter(ng, self.height, self.width)
            new_R = f.average_filter(nr, self.height, self.width)

            output_image = ( np.dstack((new_B, new_G, new_R))).astype(np.uint8)
        
        return output_image

    def gaussian_filter(self):
        if (self.imgType == 'grayscale'):
           
            output_image = f.gaussian_filter(self.noised_img, self.height, self.width)

        else:
            (nb, ng, nr) = cv2.split(self.noised_img)
            new_B = f.gaussian_filter(nb, self.height, self.width)
            new_G = f.gaussian_filter(ng, self.height, self.width)
            new_R = f.gaussian_filter(nr, self.height, self.width)

            output_image = ( np.dstack((new_B, new_G, new_R)))
        
        return output_image
    

    def sobel(self):
        if (self.imgType == 'grayscale'):
           
            output_image = f.sobel(self.imgByte, self.height, self.width)

        else:
            new_B = f.sobel(self.b, self.height, self.width)
            new_G = f.sobel(self.g, self.height, self.width)
            new_R = f.sobel(self.r, self.height, self.width)
        

            output_image =  np.dstack((new_B, new_G, new_R))
        
        return output_image

    def prewitt(self):
        if (self.imgType == 'grayscale'):
           
            output_image = f.prewitt(self.imgByte, self.height, self.width)

        else:
            new_B = f.prewitt(self.b, self.height, self.width)
            new_G = f.prewitt(self.g, self.height, self.width)
            new_R = f.prewitt(self.r, self.height, self.width)
        

            output_image =  np.dstack((new_B, new_G, new_R))
        
        return output_image

    def roberts(self):
        if (self.imgType == 'grayscale'):
           
            output_image = f.roberts(self.imgByte, self.height, self.width)

        else:
            new_B = f.roberts(self.b, self.height, self.width)
            new_G = f.roberts(self.g, self.height, self.width)
            new_R = f.roberts(self.r, self.height, self.width)
        

            output_image =  np.dstack((new_B, new_G, new_R))
        
        return output_image

    
    def otsu(self):
        if (self.imgType == 'grayscale'):
            gray = copy(self.imgByte)
            gray_flat = copy(self.imgFlat)
          
        else:
            gray = copy(self.to_grayScale())
            gray_flat = gray.flatten()

            
           
        pixel_number = gray.shape[0] * gray.shape[1]
        mean_weigth = 1.0 / pixel_number
        his, bins = f.get_histogram(gray_flat),  np.arange(256)
        
        final_thresh = 0
        final_value = 0
        for t in bins[1:-1]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
            Wb = np.sum(his[:t]) * mean_weigth
            Wf = np.sum(his[t:]) * mean_weigth

            mub = np.mean(his[:t])
            muf = np.mean(his[t:])
           
            value = Wb * Wf * (mub - muf) ** 2

           

            if value > final_value:
                final_thresh = t
                final_value = value
        final_img = gray.copy()
        
        final_img[gray > final_thresh] = 255
        final_img[gray < final_thresh] = 0
        return final_img

    def contrast_threshold(self):

        if ((self.imgType == 'grayscale')):
            img = copy(self.imgByte)
        else:
            img = copy(self.to_grayScale())

        w_size = 15
        thresholds = np.zeros(img.shape)

        # Obtaining windows
        hw_size = w_size // 2
        padded_img = np.ones((img.shape[0] + w_size - 1,
                              img.shape[1] + w_size - 1)) * np.nan
        padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

        winds = view_as_windows(padded_img, (w_size, w_size))

        # Obtaining maximums and minimums
        mins = np.nanmin(winds, axis=(2, 3))
        maxs = np.nanmax(winds, axis=(2, 3))

        min_dif = img - mins
        max_dif = maxs - img

        thresholds[min_dif <= max_dif] = 256
        thresholds[min_dif > max_dif] = 0
       

        return thresholds