import numpy as np 
import matplotlib.pyplot as plt
import cv2
from copy import copy 
from itertools import combinations
import cv2
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer
import meanshift as ms


''' Kmeans'''
def euclidean_distance(pixel ,centroid ):
    return np.sqrt(np.sum((pixel - centroid)**2))


def get_clusters(image , centroids, k):
    clusters = [[] for _ in range(k)]
    for idx, pixel in enumerate(image):

        distances = [euclidean_distance(pixel, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)

        clusters[closest_centroid_index].append(idx)

    return clusters

def get_new_centroids(clusters ,image, k , chennels_number):
    # assign mean value of clusters to centroids
    centroids = np.zeros((k, chennels_number))
    for cluster_idx, cluster in enumerate(clusters):
        cluster_mean = np.mean(image[cluster], axis=0)
        centroids[cluster_idx] = cluster_mean
    return centroids

def kmeans(k , img , max_iterations = 100):
    np.random.seed(42)
    image = copy(img)
    img_size , chennels_number = image.shape
    # choose k random pixels as cluster centers ---> centroids
    centroids_indexes = np.random.choice(img_size, k, replace=False)
    centroids = [image[idx] for idx in centroids_indexes]

    # keep iterating to get good clusters(until the clusters converge or the max number of iterations is reached)
    for iteration in range(max_iterations):
        
        # Assign image pixels to the nearest centeroids
        clusters = get_clusters(image , centroids, k) 

        new_centroids = get_new_centroids(clusters, image,k , chennels_number)

        distances = [euclidean_distance(centroids[i], new_centroids[i]) for i in range(k)]
        if (sum(distances) == 0): # the centroids converged so break 
            print("convergent")
            break
    
    new_centroids = np.uint8(new_centroids) 
    
    for cluster_idx, cluster in enumerate(clusters):
        segmented_image_value = new_centroids[cluster_idx]
        for pixel_index in cluster:
            image[pixel_index] = segmented_image_value
            

    image = np.uint8(image)
    
    return image 
    # return get_cluster_labels(clusters,img_size) , new_centroids



''' Kmeans'''

''' Otsu '''
def _get_variance(hist, c_hist, cdf, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""
    variance = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # Cumulative histogram
        weight = c_hist[t2] - c_hist[t1 - 1]

        # Region CDF
        r_cdf = cdf[t2] - cdf[t1 - 1]

        # Region mean
        r_mean = r_cdf / weight if weight != 0 else 0

        variance += weight * r_mean ** 2

    return variance


def _get_thresholds(hist, c_hist, cdf, nthrs):
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_var = 0
    opt_thresholds = None

    # Extending histograms for convenience
    c_hist = np.append(c_hist, [0])
    cdf = np.append(cdf, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing variance for the current combination of thresholds
        regions_var = _get_variance(hist, c_hist, cdf, e_thresholds)

        if regions_var > max_var:
            max_var = regions_var
            opt_thresholds = thresholds

    return opt_thresholds


def otsu_multithreshold(image=None, hist=None, nthrs=None):
    # Histogran
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    # Cumulative histograms
    c_hist = np.cumsum(hist)
    cdf = np.cumsum(np.arange(len(hist)) * hist)

    return _get_thresholds(hist, c_hist, cdf, nthrs)

def apply_multithreshold(img, thresholds):
  
    # Extending entropy and thresholds for convenience
    e_thresholds = [-1]
    e_thresholds.extend(thresholds)

    # Threshold image
    t_image = np.zeros_like(img)

    for i in range(1, len(e_thresholds)):
        t_image[img >= e_thresholds[i]] = i

    wp_val = 255 // len(thresholds)

    return t_image * wp_val

def Otsu(img):
    th = otsu_multithreshold(img, nthrs=3)
    result = apply_multithreshold(img, th)
    return result

def Otsu_local(img , mask):

    img2 = np.zeros(img.shape)
    for c in range(0, img.shape[1], mask):
        for r in range(0, img.shape[0], mask):
            window = img[r:r+mask, c:c+mask]
            img2[r:r+mask, c:c+mask] = Otsu(window)
            
    return img2




''' Otsu '''

'''Optimal Thresholding '''
def optimal_thresholding(img):
    bg_Sum = (img[0,0]+img[0,-1]+img[-1,0]+img[-1,-1])
    fg_Sum = np.sum(img) - bg_Sum
    bg_mean = bg_Sum/4
    fg_mean= fg_Sum/ (np.size(img)-4)
    t = ( bg_mean + fg_mean)/2
    while True:
        bg = img[img < t]
        fg = img[img >= t]
        if(len(bg)!= 0 and len(fg) != 0):

            bg_mean = np.mean(bg)
            fg_mean = np.mean(fg)
        else:
            pass
        if t == (bg_mean+fg_mean)/2:
            break
        t = (bg_mean+fg_mean)/2
        

        
        


    t_image = np.zeros_like(img)

    
    t_image[img >= t] = 255
    return t_image

def optimal_thresholding_local(img , mask):

    img2 = np.zeros(img.shape)
    for c in range(0, img.shape[1], mask):
        for r in range(0, img.shape[0], mask):
            window = img[r:r+mask, c:c+mask]
            img2[r:r+mask, c:c+mask] = optimal_thresholding(window)
            
    return img2
'''Optimal Thresholding '''

'''Mean shift '''
def mean_shift(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pixels = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    data = image.reshape((-1, 3))
    mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')

    mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth = [2,2,2])

    original_points =  mean_shift_result.original_points
    shifted_points = mean_shift_result.shifted_points
    cluster_assignments = mean_shift_result.cluster_ids
    segmented_image = shifted_points.reshape(image.shape)

    return segmented_image



'''Mean shift '''