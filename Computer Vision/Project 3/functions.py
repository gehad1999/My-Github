from scipy import signal
from matplotlib import cm
import matplotlib.pyplot as plt
from math import sqrt
from math import sin, cos
import cv2
import numpy as np
import scipy
import glob
import time
from copy import copy
import time
''' Harris Operator'''
# Kernel operation using input operator of size 3*3
def GetSobel(image, Sobel, width, height):
    # Initialize the matrix
    I_d = np.zeros((width, height), np.float32)

    # For every pixel in the image
    for rows in range(width):
        for cols in range(height):
            # Run the Sobel kernel for each pixel
            if rows >= 1 or rows <= width-2 and cols >= 1 or cols <= height-2:
                for ind in range(3):
                    for ite in range(3):
                        I_d[rows][cols] += Sobel[ind][ite] * image[rows - ind - 1][cols - ite - 1]
            else:
                I_d[rows][cols] = image[rows][cols]

    return I_d


def HarrisCornerDetection(image):
    t0 = time.clock()
    # The two Sobel operators - for x and y direction
    SobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    SobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    w, h = image.shape

    # X and Y derivative of image using Sobel operator
    ImgX = GetSobel(image, SobelX, w, h)
    ImgY = GetSobel(image, SobelY, w, h)

    # # Eliminate the negative values
    for ind1 in range(w):
        for ind2 in range(h):
            if ImgY[ind1][ind2] < 0:
                ImgY[ind1][ind2] *= -1
            if ImgX[ind1][ind2] < 0:
                ImgX[ind1][ind2] *= -1
                
    ImgX_2 = np.square(ImgX)
    ImgY_2 = np.square(ImgY)

    ImgXY = np.multiply(ImgX, ImgY)
    ImgYX = np.multiply(ImgY, ImgX)

    #Use Gaussian Blur
    Sigma = 1.4
    kernelsize = (3, 3)

    ImgX_2 = cv2.GaussianBlur(ImgX_2, kernelsize, Sigma)
    ImgY_2 = cv2.GaussianBlur(ImgY_2, kernelsize, Sigma)
    ImgXY = cv2.GaussianBlur(ImgXY, kernelsize, Sigma)
    ImgYX = cv2.GaussianBlur(ImgYX, kernelsize, Sigma)

    alpha = 0.06
    R = np.zeros((w, h), np.float32)
    # For every pixel find the corner strength
    for row in range(w):
        for col in range(h):
            M_bar = np.array([[ImgX_2[row][col], ImgXY[row][col]], [ImgYX[row][col], ImgY_2[row][col]]])
            R[row][col] = np.linalg.det(M_bar) - (alpha * np.square(np.trace(M_bar)))

    R_threshold = threshold(R, image.shape)

    t1 = time.clock() - t0
    return R_threshold , t1

''' End of Harris Operator'''

''' SIFT descriptor'''

def sift_gradient(img):
    dx = np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])
    dy = dx.T
    gx = signal.convolve2d( img , dx , boundary='symm', mode='same' )
    gy = signal.convolve2d( img , dy , boundary='symm', mode='same' )
    magnitude = np.sqrt( gx * gx + gy * gy )
    direction = np.rad2deg( np.arctan2( gy , gx )) % 360
    return gx,gy,magnitude,direction


def padded_slice(img, sl):
    output_shape = np.asarray(np.shape(img))
    output_shape[0] = sl[1] - sl[0]
    output_shape[1] = sl[3] - sl[2]
    src = [max(sl[0], 0),
           min(sl[1], img.shape[0]),
           max(sl[2], 0),
           min(sl[3], img.shape[1])]
    dst = [src[0] - sl[0], src[1] - sl[0],
           src[2] - sl[2], src[3] - sl[2]]
    output = np.zeros(output_shape, dtype=img.dtype)
    output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
    return output

def dog_keypoints_orientations( img , keypoints , num_bins = 36 ):
    kps = []
    
    gx,gy,magnitude,direction = sift_gradient(img)
    direction_idx = np.round( direction * num_bins / 360 ).astype(int)    
    radius = 8      
    for i,j in map( tuple , np.argwhere( keypoints ).tolist() ):
        window = [i-radius, i+radius+1, j-radius, j+radius+1]
        mag_win = padded_slice( magnitude , window ) 
        dir_idx = padded_slice( direction_idx, window )
        hist = np.zeros(num_bins, dtype=np.float32)
        for bin_idx in range(num_bins):
            hist[bin_idx] = np.sum( mag_win[ dir_idx == bin_idx ] )
        for bin_idx in np.argwhere( hist >= 0.8 * hist.max() ).tolist():
            angle = (bin_idx[0]+0.5) * (360./num_bins) % 360
            kps.append( (i,j,angle))
    return kps



def rotated_subimage(image, center, theta, width, height):
    theta *= 3.14159 / 180 # convert to rad
    
    
    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)

def kp_list_2_opencv_kp_list(kp_list):

    opencv_kp_list = []
    for kp in kp_list:
        opencv_kp = cv2.KeyPoint(x=kp[1] ,
                                 y=kp[0] ,
                                 _size=1,
                                 _angle=kp[2],
                                 )
        opencv_kp_list += [opencv_kp]

    return opencv_kp_list

def extract_sift_descriptors128( img, keypoints, num_bins = 8 ):
    descriptors = []
    points = []  
    gx,gy,magnitude,direction = sift_gradient(img)

    for (i,j, orientation) in keypoints:

        window_mag = rotated_subimage(magnitude,(j,i), orientation, 16,16)
        window_dir = rotated_subimage(direction,(j,i), orientation, 16,16)
        window_dir = (((window_dir - orientation) % 360) * num_bins / 360.).astype(int)

        features = []
        for sub_i in range(4):
            for sub_j in range(4):
                sub_weights = window_mag[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                sub_dir_idx = window_dir[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                hist = np.zeros(num_bins, dtype=np.float32)
                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum( sub_weights[ sub_dir_idx == bin_idx ] )
                features.extend( hist)
                # features.extend( hist.tolist())
        features = np.array(features) 
        features /= (np.linalg.norm(features))
        np.clip( features , np.finfo(np.float16).eps , 0.2 , out = features )
        assert features.shape[0] == 128, "features missing!"
        features /= (np.linalg.norm(features))
        descriptors.append(features)
        points.append( (i ,j , orientation))
    points = kp_list_2_opencv_kp_list(points)
    return points , np.array(descriptors)

''' End of SIFT descriptor'''




''' SSD Matcher '''
def SSDdistance_matrix(A, B, squared=False):
    """
    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

def SSDFeatureMatcher( desc1, desc2):
        matches = []
        n1 = desc1.shape[0]      
        distance = SSDdistance_matrix(desc1, desc2)
        match = np.argmin(distance, 1)
        for i in range(n1):
            f = cv2.DMatch()
            f.queryIdx = i
            f.trainIdx = int(match[i])
            f.distance = distance[i, int(match[i])]
            matches.append(f)

        return matches


''' End of SSD Matcher '''

''' Normalized Cross correlation '''
def nccmatrix(A, B):

    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1]

    numerator=A.dot(B.T)

    s=(A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    n=(B*B).sum(axis=1)*np.ones(shape=(M,1))
    denominator=(s*n)**0.5
    ncc =  numerator /denominator


    return ncc

def nccFeatureMatcher( desc1, desc2):
    matches = []
    n1 = desc1.shape[0]
    distance = nccmatrix(desc1, desc2)
    match = np.argmax(distance, 1)
    for i in range(n1):
        if (distance[i, int(match[i])]>0.86):
            f = cv2.DMatch()
            f.queryIdx = i
            f.trainIdx = int(match[i])
            f.distance = distance[i, int(match[i])]
            matches.append(f)

    return matches
  
''' End of Normalized Cross correlation '''

def threshold(key_pints,img_shape):
    features =np.zeros((img_shape), dtype=np.float32)
    
    features[key_pints>0.01*key_pints.max()]=255
    return features

def Harris_SIFT(image1_gray, image2_gray , img1 , img2 ,matcher):
        
    key_points1 , Harris_time_img1 = HarrisCornerDetection(image1_gray)
    print("Harris_time_img1: ", Harris_time_img1)

    key_points2 , Harris_time_img2 = HarrisCornerDetection(image2_gray)
    print("Harris_time_img2: ", Harris_time_img2)

    t0_descriptors1 = time.clock()
    key_points1 = dog_keypoints_orientations(image1_gray,key_points1, num_bins = 36)
    key_points1,descriptors1 = extract_sift_descriptors128(image1_gray , key_points1 , 8)
    t1_descriptors1 = time.clock() - t0_descriptors1

    print("descriptors_img1: ", t1_descriptors1)

    t0_descriptors2 = time.clock()
    key_points2 = dog_keypoints_orientations(image2_gray,key_points2, num_bins = 36)
    key_points2,descriptors2 = extract_sift_descriptors128(image2_gray , key_points2 , 8)
    t1_descriptors2 = time.clock() - t0_descriptors2

    print("descriptors_img2: ", t1_descriptors2)

    if (matcher == 'SSD'):
        t0_SSD = time.clock()
        matches=SSDFeatureMatcher(descriptors1,descriptors2)
        t1_SSD = time.clock() - t0_SSD 

        print("SSD_time: ", t1_SSD)
    else:
        t0_NCC = time.clock()
        matches = nccFeatureMatcher(descriptors1,descriptors2)

        t1_NCC = time.clock() - t0_NCC 
        print("NCC_time: ", t1_NCC)

    matches = sorted(matches, key = lambda x:x.distance)
    output = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), key_points1, cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), key_points2, matches[:100], copy(img2), flags=2)

    return output