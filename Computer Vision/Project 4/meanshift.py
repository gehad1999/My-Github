import numpy as np
import math
import sys

GROUP_DISTANCE_TOLERANCE = .1

MIN_DISTANCE = 0.000001
def multivariate_gaussian_kernel(distances, bandwidths):

    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * math.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val

def euclidean_dist(pointA, pointB):
    
    
    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)


def group_points(points):
    group_assignment = []
    groups = []
    group_index = 0
    for point in points:
        nearest_group_index = _determine_nearest_group(point, groups)
        if nearest_group_index is None:
            # create new group
            groups.append([point])
            group_assignment.append(group_index)
            group_index += 1
        else:
            group_assignment.append(nearest_group_index)
            groups[nearest_group_index].append(point)
    return np.array(group_assignment)

def _determine_nearest_group(point, groups):
    nearest_group_index = None
    index = 0
    for group in groups:
        distance_to_group = _distance_to_group(point, group)
        if distance_to_group < GROUP_DISTANCE_TOLERANCE:
            nearest_group_index = index
        index += 1
    return nearest_group_index

def _distance_to_group(point, group):
    min_distance = sys.float_info.max
    for pt in group:
        dist = euclidean_dist(point, pt)
        if dist < min_distance:
            min_distance = dist
    return min_distance






class MeanShift(object):
    
    def __init__(self, kernel=multivariate_gaussian_kernel):
        if kernel == 'multivariate_gaussian':
            kernel = multivariate_gaussian_kernel
        self.kernel = kernel
    
    
    def cluster(self, points, kernel_bandwidth, iteration_callback=None):
        if(iteration_callback):
            iteration_callback(points, 0)
        shift_points = np.array(points)
        # max_min_dist = 100000
        max_min_dist = 1
        iteration_number = 0

        still_shifting = [True] * points.shape[0]
        while ((max_min_dist > MIN_DISTANCE) ):
        # while ((max_min_dist > MIN_DISTANCE) and iteration_number<1000):
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
            
                
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                p_new = self._shift_point(p_new, points, kernel_bandwidth)
                dist = euclidean_dist(p_new, p_new_start)
                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new #leave
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
            
            
        # point_grouper = pg.PointGrouper()
        # group_assignments = point_grouper.group_points(shift_points.tolist())
        group_assignments = group_points(shift_points.tolist())
        return MeanShiftResult(points, shift_points, group_assignments)
    
    
    def _shift_point(self, point, points, kernel_bandwidth):
        # from http://en.wikipedia.org/wiki/Mean-shift
        points = np.array(points)

        # numerator
        point_weights = self.kernel(point-points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        # denominator
        denominator = np.sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        return shifted_point


class MeanShiftResult:
    def __init__(self, original_points, shifted_points, cluster_ids):
        self.original_points = original_points
        self.shifted_points = shifted_points
        self.cluster_ids = cluster_ids