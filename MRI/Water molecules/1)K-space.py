import matplotlib as plt
import cv2 as cv
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn

def transform_image_to_kspace(img, dim=None, k_shape=None):
#""" Computes the Fourier transform from image space to k-space space
#along a given or all dimensions
#:param img: image space data
#:param dim: vector of dimensions to transform
#:param k_shape: desired shape of output k-space data
#:returns: data in k-space (along transformed dimensions)
#"""
   if not dim:
     dim = range(img.ndim)

   k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
   k/= np.sqrt(np.prod(np.take(img.shape, dim)))
   return k
image = cv.imread("mri.jpeg",0)
kspace = transform_image_to_kspace(image)
kspace = np.asarray(kspace)
kspace = kspace.astype(np.uint8)
print(kspace)
cv.imshow("K-Space",kspace)
cv.waitKey(0)
cv.destroyAllWindows()