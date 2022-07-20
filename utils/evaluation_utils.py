import math
import numpy as np

def computeDetectionPerformanceAndLocalizationError(pts_t_gt, pts_t_hat):

    if len(pts_t_gt) == 0:
        print("no ground truth points, can't compute an error")
        return

    detection_radius_pixels = 3

    nr_points = len(pts_t_gt)
    checked_point_indices = np.zeros(len(pts_t_hat))
    nr_detections = 0
    euclidean_distances = []
    for pt_gt in pts_t_gt:
        smallest_euclidean_distance = 1000000
        best_matching_index = -1
        for idx, matching_pt in enumerate(pts_t_hat):
            if checked_point_indices[idx] == 0:
                euclidean_distance = math.sqrt((pt_gt[0] - matching_pt[0]) ** 2 + (pt_gt[1] - matching_pt[1]) ** 2)
                if euclidean_distance < detection_radius_pixels and euclidean_distance < smallest_euclidean_distance:
                    smallest_euclidean_distance = euclidean_distance
                    best_matching_index = idx

        if best_matching_index != -1:
            nr_detections += 1
            checked_point_indices[best_matching_index] = 1 # point was consumed, won't be checked again
            euclidean_distances.append(smallest_euclidean_distance)


    print(f"detected {nr_detections} points out of total {nr_points}, percentage = {nr_detections/nr_points*100}%")
    if len(euclidean_distances) > 0:
        #print(euclidean_distances)
        print(f"mean localization error: {np.mean(np.array(euclidean_distances))}")