import cv2
import numpy as np
from random import randint
from copy import copy


def get_histogram(img):
    
    histogram = np.zeros(256, dtype=int)
    # histogram = np.zeros(2^24, dtype=int)
   
    for i in range(img.size):
        histogram[img[i]] += 1

    return histogram

def get_cumsum(histogram): ### get the cumulative sum of the histogram

    # cumsum = np.zeros(2^24, dtype=int)
    cumsum = np.zeros(256, dtype=int)
    cumsum[0] = histogram[0]
    for i in range(1, histogram.size):
        cumsum[i] = cumsum[i-1] + histogram[i]
    return cumsum

def cumulative_crve(data):
    histogram = get_histogram(data)
    cum_sum = get_cumsum(histogram)
    return cum_sum

def get_mapping(cumsum, IMG_H,IMG_W):
    """ Create a mapping s.t. each old colour value is mapped to a new
        one between 0 and 255. Mapping is created using:
         - M(i) = max(0, round((grey_levels*cumsum(i))/(h*w))-1)
        where g_levels is the number of grey levels in the image """
    # S_k = np.zeros(2^24, dtype=int)
    S_k = np.zeros(256, dtype=int)
    grey_levels = 256
    # grey_levels = 2^24

    for i in range(grey_levels):
        S_k[i] = round( ( (grey_levels-1) / (IMG_H*IMG_W) ) * cumsum[i] ) ### new intensity value

    return S_k

def Apply_mapping(img, mapping):
    """ Apply the mapping to our image """
    new_image = np.zeros(img.size, dtype=int)
    for i in range(img.size):
        new_image[i] = mapping[img[i]]
    return new_image



def equalizeHist(data , height, width):
    histogram = get_histogram(data)
    cumsum = get_cumsum(histogram)
    mapping = get_mapping(cumsum, height,width )
    new_image = Apply_mapping(data, mapping)
    new_image = np.uint8(new_image.reshape((height, width )))

    return new_image


def normalize_img(data , height, width):
    min_intensity = data.min()
    max_intensity = data.max()
    normalized_img = np.zeros(data.size, dtype=int)

    for i in range(data.size):

        # normalized_img[i] = round( (2^24)-1 * ( (img[i] - min_intensity) / (max_intensity - min_intensity)))
        normalized_img[i] = round( 255 * ( (data[i] - min_intensity) / (max_intensity - min_intensity)))
    normalized_img = np.uint8(normalized_img.reshape((height,width)))
    return normalized_img

def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base


def freq_lowpass_Filter(data):
    img_fft = np.fft.fft2(data)
    img_fftshift = np.fft.fftshift(img_fft)


    LowPassCenter = img_fftshift * gaussianLP(20,data.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)

    return (np.abs(inverse_LowPass)).astype(np.uint8)

def freq_highpass_Filter(data):


    img_fft = np.fft.fft2(data)
    img_fftshift = np.fft.fftshift(img_fft)


    HighPassCenter = img_fftshift * gaussianHP(20,data.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)

    return np.uint8(np.abs(inverse_HighPass))

def salt_and_pepper_noise(img,height,width):
    data = copy(img)
    noise_percent = 10
    for i in range (data.size):
    
        noise_check = randint(1,10)
        if noise_check == noise_percent :
            noise_value = randint(0,255)
            data[i] = noise_value

    noise_img = np.uint8(data.reshape((height,width)))

    return noise_img

def median_filter(img, height, width):
    img_new = np.zeros([height, width])
    for i in range(1, height-1):
        for j in range(1,  width-1):
            temp = [img[i-1, j-1], img[i-1, j], img[i-1, j + 1], img[i, j-1],img[i, j],
                img[i, j + 1],img[i + 1, j-1], img[i + 1, j], img[i + 1, j + 1]]
            temp = sorted(temp)
            img_new[i, j]= temp[4]

    return img_new

def average_filter(img, height, width):
    img_new = np.zeros([height, width])
  
    for i in range(1, height-1):
        for j in range(1, width-1):
         
            temp = [img[i-1, j-1], img[i-1, j], img[i-1, j + 1], img[i, j-1],img[i, j],
                img[i, j + 1],img[i + 1, j-1], img[i + 1, j], img[i + 1, j + 1]]

            img_new[i, j]= (np.asarray(temp)).sum() / 9   
    
    
    return img_new

def gaussian_filter(img, height, width):
    img_new = np.zeros([height,  width])

    mask = np.ones([3, 3], dtype = int)
    mask[0,1]=2
    mask[1,0]=2
    mask[1,1]=4
    mask[1,2]=2
    mask[2,1]=2
    
    mask = mask / 16

    for i in range(1, height-1):
        for j in range(1, width-1):
            temp = img[i-1, j-1]*mask[0, 0]+\
            img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+\
            img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+\
            img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+\
            img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

            img_new[i, j]= temp

    img_new= img_new.astype(np.uint8)

    return img_new
    
def sobel(img, rows, columns):
  
    sobel_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    sobel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
  
    sobel_filtered_image = np.zeros(shape=(rows, columns))
    for i in range(rows - 2):
        for j in range(columns - 2):
            sobelX = np.sum(np.multiply(sobel_x, img[i:i + 3, j:j + 3]))
            sobelY = np.sum(np.multiply(sobel_y, img[i:i + 3, j:j + 3]))
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(sobelX ** 2 + sobelY ** 2)
    return sobel_filtered_image.astype(np.uint8)

def prewitt(img, rows, columns):
    
    prewitt_x = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    prewitt_y = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    
    prewitt_filtered_image = np.zeros(shape=(rows, columns))
    for i in range(rows - 2):
        for j in range(columns - 2):
            prewittX = np.sum(np.multiply(prewitt_x, img[i:i + 3, j:j + 3]))
            prewittY = np.sum(np.multiply(prewitt_y, img[i:i + 3, j:j + 3]))
            prewitt_filtered_image[i + 1, j + 1] = np.sqrt(prewittX ** 2 + prewittY ** 2)
    return prewitt_filtered_image.astype(np.uint8)

def roberts(img, rows, columns):

    roberts_x = np.array([[1.0, 0.0], [0.0, -1.0]])
    roberts_y = np.array([[0.0, -1.0], [1.0, 0.0]])
    roberts_filtered_image = np.zeros(shape=(rows, columns))
    for i in range(rows - 2):
        for j in range(columns - 2):
            robertsX = np.sum(np.multiply(roberts_x, img[i:i + 2, j:j + 2]))
            robertsY = np.sum(np.multiply(roberts_y, img[i:i + 2, j:j + 2]))
            roberts_filtered_image[i + 1, j + 1] = np.sqrt(robertsX ** 2 + robertsY ** 2)
    return roberts_filtered_image.astype(np.uint8)

