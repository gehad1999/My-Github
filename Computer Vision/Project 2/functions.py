import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from itertools import product
from copy import copy


''' Start Canny Edge detection '''
def convolution(image, kernel ):
    
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    output = np.zeros(image.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
     

    return output


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(image, kernel)



def sobel_edge_detection(img):
        
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolution(img, sobel_x)
    Iy = convolution(img, sobel_y)

    gradient_magnitude = np.sqrt(np.square(Ix) + np.square(Iy))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()   
    gradient_direction = np.arctan2(Iy, Ix)   
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction += 180

    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape

    output = np.zeros(gradient_magnitude.shape)

    PI = 180

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]

            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]

            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]

    return output

def double_threshold(image, lowThreshold, highThreshold, weak): #5,20
    output = np.zeros(image.shape)

    strong = 255

    strong_row, strong_col = np.where(image >= highThreshold)
    weak_row, weak_col = np.where((image <= highThreshold) & (image >= lowThreshold))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    

    return output

def hysteresis(image, weak):
    image_row, image_col = image.shape

    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[
                    row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                    row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[
                    row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                    row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[
                    row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                    row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[
                    row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                    row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image

def canny_edge_detection(image):
    
    blurred_image = gaussian_blur(image, kernel_size=9)  
    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image )  
    new_image = non_max_suppression(gradient_magnitude, gradient_direction)
    weak = 50
    new_image = double_threshold(new_image, 5, 20, weak=weak)
    new_image = hysteresis(new_image, weak)

    return new_image

'''End Canny Edge detection'''

'''Hough transform line detection'''
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    ''' A function for creating a Hough Accumulator for lines in an image. '''
    height, width = img.shape # we need heigth and width to calculate the diag
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas



def hough_peaks(H, num_peaks, threshold=0, nhood_size=20):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H




# drawing the lines from the Hough Accumulatorlines using OpevCV cv2.line
def hough_lines(indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    lines = []
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        lines.append([x1,x2, y1,y2])
       
    return lines    

def hough_transform_lines(image):
    edge_image = canny_edge_detection(image)
    H, rhos, thetas = hough_lines_acc(edge_image)
    indicies, H = hough_peaks(H,5, nhood_size=15) # find peaks
    # plot_hough_acc(H) # plot hough space, brighter spots have higher votes
    # hough_lines_draw(shapes, indicies, rhos, thetas)
    lines = hough_lines( indicies, rhos, thetas)

    return lines

'''End Hough transform line detection'''


'''Hough transform Circle detection'''

def detectCircles(img,threshold,region,radius = None):
    (M,N) = img.shape
    if radius == None:
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius

    R = R_max - R_min
    #Initializing accumulator array.
    #Accumulator array is a 3 dimensional array with the dimensions representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                               #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X = [x-m+R_max,x+m+R_max]                                           #Computing the extreme X values
            Y= [y-n+R_max,y+n+R_max]                                            #Computing the extreme Y values
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0

    for r,x,y in np.argwhere(A):
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]
        try:
            p,a,b = np.unravel_index(np.argmax(temp),temp.shape)
        except:
            continue
        B[r+(p-region),x+(a-region),y+(b-region)] = 1

    return B[:,R_max:-R_max,R_max:-R_max]

def displayCircles(A ):
    # img = image
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.imshow(img)
    circleCoordinates = np.argwhere(A)                                          #Extracting the circle information
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False , linewidth=3))
        # ax.Circle((y,x),r,color=(1,0,0),fill=False)
        # ax.add_artist(circle[-1])
    # print(circle)
    # plt.show()
    return circle



def hough_transform_circles(image):
    edge_image = canny_edge_detection(image)
    res = detectCircles(edge_image,12,15,radius=[100,10])
    
    
    circles = displayCircles(res)

    return circles   
'''End Hough transform Circle detection'''

'''Active Contours'''
# _ALPHA = 170
_ALPHA = 100
# _BETA = 90
_BETA = 200
_W_LINE = 250
_W_EDGE = 30
_MIN_DISTANCE = 10
_INITIAL_SMOOTH = 15
# _INITIAL_ITERATIONS = 70
_INITIAL_ITERATIONS = 30
_ITERATIONS_DELTA = 5 
_SMOOTH_FACTOR_DELTA = 4
_NUM_NEIGHBORS = 9
_MAX_SNAXELS = 10000
_INITIAL_DISTANCE_BETWEEN_SNAXELS = 50

def _gradientImage(image):
    """
    Obtain a gradient image (in both x and y directions)
    """
    gradient = np.sqrt(filters.sobel(image, 0)**2 + filters.sobel(image, 1)**2)
    gradient -= gradient.min()

    return gradient 

def _inBounds(image, point):
    """
    Is the point within the bounds of the image?
    """
    return np.all(point < np.shape(image)) and np.all(point > 0)

def _externalEnergy(image, smooth_image, point):
    """
    The external energy of the point, a combination of line and edge 
    """
    pixel = 255 * image[point[1]][point[0]]
    smooth_pixel = 255 * smooth_image[point[1]][point[0]]
    external_energy = (_W_LINE * pixel) - (_W_EDGE * (smooth_pixel**2))
    return external_energy

def _energy(image, smooth_image, current_point, next_point, previous_point=None):
    """
    Total energy (internal and external).
    Internal energy measures the shape of the contour
    """
    d_squared = np.linalg.norm(next_point -current_point)**2
    
    if previous_point is None:
        e =  _ALPHA * d_squared + _externalEnergy(image, smooth_image, current_point)
        return e 
    else:
        deriv = np.sum((next_point - 2 * current_point + previous_point)**2)
        e = 0.5 * (_ALPHA * d_squared + _BETA * deriv + _externalEnergy(image, smooth_image, current_point))
        return e

def _iterateContour(image, smooth_image, snaxels, energy_matrix, position_matrix, neighbors):
    """
    Compute the minimum energy locations for all the snaxels in the contour
    """
    snaxels_added = len(snaxels)
    for curr_idx in range(snaxels_added - 1, 0, -1):
        energy_matrix[curr_idx][:][:] = float("inf")
        prev_idx = (curr_idx - 1) % snaxels_added
        next_idx = (curr_idx + 1) % snaxels_added
        
        for j, next_neighbor in enumerate(neighbors):
            next_node = snaxels[next_idx] + next_neighbor
            
            if not _inBounds(image, next_node):
                continue
            
            min_energy = float("inf")
            for k, curr_neighbor in enumerate(neighbors):
                curr_node = snaxels[curr_idx] + curr_neighbor
                distance = np.linalg.norm(next_node - curr_node)
                
                if not _inBounds(image, curr_node) or (distance < _MIN_DISTANCE):
                    continue
                
                
                min_energy = float("inf")
                for l, prev_neighbor in enumerate(neighbors):
                    prev_node = snaxels[prev_idx] + prev_neighbor
                        
                    if not _inBounds(image, prev_node):
                        continue
                        
                    energy = energy_matrix[prev_idx][k][l] + _energy(image, smooth_image, curr_node, next_node, prev_node)
                    
                    if energy < min_energy:
                        x=1
                        min_energy = energy
                        min_position_k = k
                        min_position_l = l
                    
                
                energy_matrix[curr_idx][j][k] = min_energy
                position_matrix[curr_idx][j][k][0] = min_position_k
                position_matrix[curr_idx][j][k][1] = min_position_l
    
    min_final_energy = float("inf")
    min_final_position_j = 0
    min_final_position_k = 0

    for j in range(_NUM_NEIGHBORS):
        for k in range(_NUM_NEIGHBORS):
            if energy_matrix[snaxels_added - 2][j][k] < min_final_energy:
                min_final_energy = energy_matrix[snaxels_added - 2][j][k]
                min_final_position_j = j
                min_final_position_k = k

    pos_j = min_final_position_j
    pos_k = min_final_position_k
    
    for i in range(snaxels_added - 1, -1, -1):
        snaxels[i] = snaxels[i] + neighbors[pos_j]
        if i > 0:
            pos_j = position_matrix[i - 1][pos_j][pos_k][0]
            pos_k = position_matrix[i - 1][pos_j][pos_k][1]
            
    return min_final_energy

def activeContour(image, snaxels):
    """
    Iterate the contour until the energy reaches an equilibrium
    """
    energy_matrix = np.zeros( (_MAX_SNAXELS - 1, _NUM_NEIGHBORS, _NUM_NEIGHBORS), dtype=np.float32)
    position_matrix = np.zeros( (_MAX_SNAXELS - 1, _NUM_NEIGHBORS, _NUM_NEIGHBORS, 2), dtype=np.int32 )
    neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])
    min_final_energy_prev = float("inf")
    
    counter = 0
    smooth_factor = _INITIAL_SMOOTH 
    iterations = _INITIAL_ITERATIONS
    gradient_image = _gradientImage(image)
    smooth_image = cv2.blur(gradient_image, (smooth_factor, smooth_factor))
        
    while True:
        counter += 1
        if not (counter % iterations):
            iterations += _ITERATIONS_DELTA
            if smooth_factor > _SMOOTH_FACTOR_DELTA:
                smooth_factor -= _SMOOTH_FACTOR_DELTA            
            smooth_image = cv2.blur(gradient_image, (smooth_factor, smooth_factor))
            print ("Deblur step, smooth factor now: ", smooth_factor)
        
        # _display(smooth_image, snaxels)
        min_final_energy = _iterateContour(image, smooth_image, snaxels, energy_matrix, position_matrix, neighbors)
        
        if (min_final_energy == min_final_energy_prev) or smooth_factor < _SMOOTH_FACTOR_DELTA:
            print ("Min energy reached at ", min_final_energy)
            print ("Final smooth factor ", smooth_factor)
            break
        else:
            min_final_energy_prev = min_final_energy

def _pointsOnCircle(center, radius, num_points=12):
    points = np.zeros((num_points, 2), dtype=np.int32)
    for i in range(num_points):
        theta = float(i)/num_points * (2 * np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        p = [x, y]
        points[i] = p
        
    return points


def activeContourFromCircle(image, center, radius):
        

    if image.ndim > 2:
        image = np.mean(image, axis=2)
       
    num_points = int((2 * np.pi * radius)/_INITIAL_DISTANCE_BETWEEN_SNAXELS)
    snaxels = _pointsOnCircle(center, radius, 30)
    old_snaxels = copy(snaxels)
    activeContour(image, snaxels)
    

    return (snaxels , old_snaxels )
'''End of Active Contours'''

