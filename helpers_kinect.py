# import cv2
# import numpy as np
# from collections import defaultdict
# import math
# import random
# import ast

# # This is a file which contains all the helper functions

# ### start function definitions###

# # find the two largest rectangles and give the length from one end of one to the other


# def findLargestCurrRect(rect_dict):
#     max_area = -1
#     # initializes largest_rect to be the first rect in the keys array of rect_dict
#     largest_rect = next(iter(rect_dict))
#     for rect in rect_dict.keys():
#         dims = rect[1]  # width and height of curr rect
#         w = dims[0]
#         h = dims[1]
#         area = w * h
#         if area > max_area:
#             max_area = area
#             largest_rect = rect

#     rect = largest_rect
#     box = rect_dict[rect]
#     return rect, box

# # returns the distance between two points


# def distance(x1, y1, x2, y2):
#     # Calculate the distance between the two points
#     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     # Return the distance
#     return dist

# # find the endpoints of the line to be drawn. In theory, these will define the 2D boundary of the rod.


# def findEndPointsLine(wood_box, color_box):
#     # midpoint of top-left and bottom-left of rectangle
#     midpoint_left_wood_box = (wood_box[0]+wood_box[3])//2
#     # midpoint of top-right and bottom-right of rectangle
#     midpoint_right_wood_box = (wood_box[1]+wood_box[2])//2
#     # midpoint of top-left and bottom-left of rectangle
#     midpoint_left_color_box = (color_box[0]+color_box[3])//2
#     # midpoint of top-right and bottom-right of rectangle
#     midpoint_right_color_box = (color_box[1]+color_box[2])//2
#     dist_lw_ro = distance(midpoint_left_wood_box[0], midpoint_left_wood_box[1],
#                           midpoint_right_color_box[0], midpoint_right_color_box[1])
#     dist_lo_rw = distance(midpoint_left_color_box[0], midpoint_left_color_box[1],
#                           midpoint_right_wood_box[0], midpoint_right_wood_box[1])
#     endpoints = []
#     if (dist_lw_ro > dist_lo_rw):  # wood is on the left
#         endpoints = [midpoint_left_wood_box, midpoint_right_color_box]
#     else:
#         endpoints = [midpoint_left_color_box, midpoint_right_wood_box]
#     return endpoints


# def findEndPointsLineAndCorners(wood_box, color_box):
#     # midpoint of top-left and bottom-left of rectangle
#     midpoint_left_wood_box = (wood_box[0]+wood_box[3])//2
#     # midpoint of top-right and bottom-right of rectangle
#     midpoint_right_wood_box = (wood_box[1]+wood_box[2])//2
#     # midpoint of top-left and bottom-left of rectangle
#     midpoint_left_color_box = (color_box[0]+color_box[3])//2
#     # midpoint of top-right and bottom-right of rectangle
#     midpoint_right_color_box = (color_box[1]+color_box[2])//2
#     dist_lw_ro = distance(midpoint_left_wood_box[0], midpoint_left_wood_box[1],
#                           midpoint_right_color_box[0], midpoint_right_color_box[1])
#     dist_lo_rw = distance(midpoint_left_color_box[0], midpoint_left_color_box[1],
#                           midpoint_right_wood_box[0], midpoint_right_wood_box[1])
#     endpoints = []
#     corners = []
#     if (dist_lw_ro > dist_lo_rw):  # wood is on the left
#         endpoints = [midpoint_left_wood_box, midpoint_right_color_box]
#         corners = [wood_box[0], wood_box[3], color_box[1], color_box[2]]
#     else:
#         endpoints = [midpoint_left_color_box, midpoint_right_wood_box]
#         corners = [color_box[0], color_box[3], wood_box[1], wood_box[2]]
#     return endpoints, corners


# def is_negative():
#     rand_int = random.randint(0, 1)
#     rand_sign = -1 if rand_int == 0 else 1
#     return rand_sign

# # returns the minimum distance rows from between two rows of a bounding box


# def minDistRows(boxPoints):
#     minDist = float('inf')
#     point1row = boxPoints[0][0]
#     point2row = boxPoints[1][0]
#     point3row = boxPoints[2][0]
#     point4row = boxPoints[3][0]
#     rows = [point1row, point2row, point3row, point4row]
#     row1idx = 0
#     row2idx = 0
#     for i in range(0, len(rows)-1):
#         for j in range(i+1, len(rows)):
#             if rows[i] == rows[j]:
#                 continue
#             dist = abs(rows[i] - rows[j])
#             if dist < minDist:
#                 minDist = dist
#                 row1idx = i
#                 row2idx = j
#     return [rows[row1idx], rows[row2idx]]

# # returns the minimum distance columns from between two columns of a bounding box


# def minDistCols(boxPoints):
#     minDist = float('inf')
#     point1col = boxPoints[0][1]
#     point2col = boxPoints[1][1]
#     point3col = boxPoints[2][1]
#     point4col = boxPoints[3][1]
#     cols = [point1col, point2col, point3col, point4col]
#     col1idx = 0
#     col2idx = 0
#     for i in range(0, len(cols)-1):
#         for j in range(i+1, len(cols)):
#             if cols[i] == cols[j]:
#                 continue
#             dist = abs(cols[i] - cols[j])
#             if dist < minDist:
#                 minDist = dist
#                 col1idx = i
#                 col2idx = j
#     return [cols[col1idx], cols[col2idx]]


# # converts a text file that contains a dictionary to a usable Python dictionary


# def txt_to_dict(filename):
#     # Open the file
#     with open(filename, 'r') as file:
#         # Initialize an empty dictionary
#         my_dict = {}
#         # Iterate over each line in the file
#         for line in file:
#             # Split the line into key and value using the colon separator
#             line = line.strip('\n')
#             key, value = line.split(':')
#             # Add the key value pair to the dictionary
#             key = ast.literal_eval(key)
#             value = ast.literal_eval(value)
#             my_dict[key] = value
#     return my_dict

# # calculates the distance between two points


# def eucl_dist(point1, point2):
#     x1, y1, z1 = point1
#     x2, y2, z2 = point2
#     distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
#     return distance


# def separate_into_color_regions(d):  # return 2 lists of points: red and blue
#     red_points = []
#     blue_points = []

#     for key in d:
#         curr = d[key]
#         if curr[0] == 0.0:  # red
#             red_points.append(key)
#         else:  # blue
#             blue_points.append(key)
#     return red_points, blue_points


# def point_cloud_centroid(point_cloud):
#     arr = point_cloud
#     sum_x = 0
#     sum_y = 0
#     sum_z = 0

#     if len(point_cloud) == 0:
#         return [0, 0, 0]

#     num_points = 0
#     for point in arr:
#         sum_x += point[0]
#         sum_y += point[1]
#         sum_z += point[2]
#         num_points += 1

#     avg_x = sum_x/num_points
#     avg_y = sum_y/num_points
#     avg_z = sum_z/num_points

#     return [avg_x, avg_y, avg_z]

# # return 1 dict of points: red and blue. Uses distance from centroid.


# def eliminate_outliers_dict_dist_centroid(d):
#     red_points, blue_points = separate_into_color_regions(d)
#     # farthest possible distance between two points from the centroid of the color region
#     max_dist_in_mm = 27.29

#     # find centroid of red points
#     red_centroid = point_cloud_centroid(red_points)
#     # find centroid of blue points
#     blue_centroid = point_cloud_centroid(blue_points)

#     for point in red_points:
#         if eucl_dist(point, red_centroid) > max_dist_in_mm:
#             del d[point]

#     for point in blue_points:
#         if eucl_dist(point, blue_centroid) > max_dist_in_mm:
#             del d[point]

#     return d

# # finds the two points in the color region that are farthest apart and stores them in an array


# def farthest_points_in_color_region(color_pts_arr):
#     max_dist = float('-inf')
#     max_dist_pts = []
#     for i in range(len(color_pts_arr)):
#         for j in range(i+1, len(color_pts_arr)):
#             dist = eucl_dist(color_pts_arr[i], color_pts_arr[j])
#             if dist > max_dist:
#                 max_dist = dist
#                 max_dist_pts = [color_pts_arr[i], color_pts_arr[j]]
#     return max_dist_pts


# def count_red_pts(d):
#     count = 0
#     for key in d:
#         curr = d[key]
#         if curr[0] == 0.0:
#             count += 1
#     return count


# def count_blue_pts(d):
#     count = 0
#     for key in d:
#         curr = d[key]
#         if curr[0] == (240.0 / 360.0):
#             count += 1
#     return count

# ### end function definitions###
