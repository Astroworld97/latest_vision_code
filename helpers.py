# helper functions
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import eig
import random
import colorsys
import ast

# Below are all the ICP helper functions
### start ICP function definitions###


def create_random_quat():
    # random.seed(22)
    random_angle = random.randint(0, 360)
    real_component = math.cos(random_angle/2)
    i_hat = math.sin(random_angle/2)
    j_hat = math.sin(random_angle/2)
    k_hat = math.sin(random_angle/2)
    # real_component = random.random()
    # i_hat = random.random()
    # j_hat = random.random()
    # k_hat = random.random()
    # return [real_component, i_hat, j_hat, k_hat]
    # real_component = random.randint(1, 10)
    # i_hat = random.randint(1, 10)
    # j_hat = random.randint(1, 10)
    # k_hat = random.randint(1, 10)
    return [real_component, i_hat, j_hat, k_hat]
    # return [-0.7240972, 0, 0, -0.6896979] #234 deg rotation about z
    # return [0.707, 0, 0.707, 0] #90 deg rotation about y
    # return [0.707, 0, 0.707, 0] #90 deg rotation about y
    # return [-0.8733046, -0.4871745, 0, 0] #45 deg rotation about x


def create_random_translation_vector():
    # random.seed(14)
    i_hat = random.randint(1, 5)
    j_hat = random.randint(1, 5)
    k_hat = random.randint(1, 5)
    return [i_hat, j_hat, k_hat]


def quat_norm(quat):  # returns the norm of the quaternion
    norm = np.sqrt(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
    return norm


# buggy: only applies translation in its current state
def apply_initial_translation_and_rotation(point_cloud, colorDict):
    quat = create_random_quat()
    norm = quat_norm(quat)
    quat = [quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm]
    quat_star = quat_conjugate(quat)
    p_centroid = point_cloud_centroid(point_cloud)
    translation_vector = create_random_translation_vector()

    for i in range(len(point_cloud)):
        point = point_cloud[i]
        color = colorDict[tuple(point)]
        colorDict[tuple(point)] = ()
        left = quat_mult(quat, point)
        right = quat_mult(left, quat_star)
        rotation = right
        moved_point = [rotation[1] + translation_vector[0], rotation[2] +
                       translation_vector[1], rotation[3] + translation_vector[2]]
        # moved_point = [point[0] + translation_vector[0], point[1] + translation_vector[1], point[2] + translation_vector[2]]#
        point_cloud[i] = moved_point
        colorDict[tuple(moved_point)] = color
    return point_cloud, colorDict


# inputs are quaternions. Remember the 0th element is irrelevant here.
def quat_dot_product(p, q):
    if len(p) == 3:
        p = vect_to_quat(p)
    if len(q) == 3:
        q = vect_to_quat(q)
    return p[1]*q[1] + p[2]*q[2] + p[3]*q[3]  # returns a scalar


# inputs are quaternions. Remember the 0th element is irrelevant here.
def quat_cross_product(p, q):
    if len(p) == 3:
        p = vect_to_quat(p)
    if len(q) == 3:
        q = vect_to_quat(q)
    i_hat = p[2]*q[3] - p[3]*q[2]
    j_hat = p[3]*q[1] - p[1]*q[3]
    k_hat = p[1]*q[2] - p[2]*q[1]
    return [i_hat, j_hat, k_hat]  # returns a vector


# inputs are quaternions. Calculates the conjugate of a quaternion.
def quat_conjugate(quat):
    first_term = quat[0]
    i_hat = -quat[1]
    j_hat = -quat[2]
    k_hat = -quat[3]
    return [first_term, i_hat, j_hat, k_hat]


def vect_to_quat(vect):  # adds a zero for the real part to turn a vector into a quaternion
    return [0, vect[0], vect[1], vect[2]]


# extracts the vector part from a quaternion.
def extract_vect_from_quat(quat):
    return [quat[1], quat[2], quat[3]]


# quaternion multiplication. Inputs p and q are quaternions.
def quat_mult(p, q):
    if len(p) == 3:
        p = vect_to_quat(p)
    if len(q) == 3:
        q = vect_to_quat(q)
    first_term = p[0]*q[0]
    dot_prod = quat_dot_product(p, q)
    first_term = first_term - dot_prod
    cross_prod = quat_cross_product(p, q)
    i_hat = p[0]*q[1] + q[0]*p[1] + cross_prod[0]
    j_hat = p[0]*q[2] + q[0]*p[2] + cross_prod[1]
    k_hat = p[0]*q[3] + q[0]*p[3] + cross_prod[2]
    return [first_term, i_hat, j_hat, k_hat]


def create_prime_matrix_p(prime_vector):
    prime_vector = [0, prime_vector[0], prime_vector[1], prime_vector[2]]
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, prime_vector[3], -1*prime_vector[2],
              prime_vector[2], -1*prime_vector[3], 0, prime_vector[1], prime_vector[3], prime_vector[2], -1*prime_vector[1], 0]
    values = np.array(values).reshape((4, 4))
    # values = list(values)
    # for i in range(len(values)):
    #     values[i] = list(values[i])
    return values


def create_prime_matrix_q(prime_vector):
    prime_vector = [0, prime_vector[0], prime_vector[1], prime_vector[2]]
    values = [0, -1*prime_vector[1], -1*prime_vector[2], -1*prime_vector[3], prime_vector[1], 0, -1*prime_vector[3], prime_vector[2],
              prime_vector[2], prime_vector[3], 0, -1*prime_vector[1], prime_vector[3], -1*prime_vector[2], prime_vector[1], 0]
    values = np.array(values).reshape((4, 4))
    # values = list(values)
    # for i in range(len(values)):
    #     values[i] = list(values[i])
    return values


# computes centroid/mean/center of mass of point cloud
def point_cloud_centroid(point_cloud):
    arr = point_cloud
    sum_x = 0
    sum_y = 0
    sum_z = 0

    if len(point_cloud) == 0:
        return [0, 0, 0]

    num_points = 0
    for point in arr:
        sum_x += point[0]
        sum_y += point[1]
        sum_z += point[2]
        num_points += 1

    avg_x = sum_x/num_points
    avg_y = sum_y/num_points
    avg_z = sum_z/num_points

    return [avg_x, avg_y, avg_z]


def calc_single_prime(point, centroid):
    prime_x = point[0] - centroid[0]
    prime_y = point[1] - centroid[1]
    prime_z = point[2] - centroid[2]
    return [prime_x, prime_y, prime_z]


def calc_single_M(P, Q):
    P_transpose = P.T
    retval = np.matmul(P_transpose, Q)
    return retval


def calc_quat(M):
    eigenvalues, eigenvectors = eig(M)
    max_eigval_idx = np.argmax(eigenvalues)
    max_eigvec = eigenvectors[:, max_eigval_idx]
    return max_eigvec


def calc_b(q_centroid, p_centroid, quat, quat_star):
    first_mult = quat_mult(quat, p_centroid)
    second_mult = quat_mult(first_mult, quat_star)
    vect = [second_mult[1], second_mult[2], second_mult[3]]
    retval = [q_centroid[0]-vect[0], q_centroid[1] -
              vect[1], q_centroid[2]-vect[2]]
    return retval


# creates the match dictionary, which associates each point in arr1 to the point closest to in arr2
def createMatchDictionary(point_cloud_p, matchDict):
    matchDict.clear()
    for point in point_cloud_p:
        matchDict[tuple(point)] = None


def color_match(point_color, cylpoint_color):
    if abs(point_color[0] - cylpoint_color[0]) <= .1 and abs(point_color[1] - cylpoint_color[1]) <= .1 and abs(point_color[2] - cylpoint_color[2]) <= .1:
        return True
    else:
        return False


def txt_to_dict(filename):
    # Open the file
    with open(filename, 'r') as file:
        # Initialize an empty dictionary
        my_dict = {}
        # Iterate over each line in the file
        for line in file:
            # Split the line into key and value using the colon separator
            line = line.strip('\n')
            key, value = line.split(':')
            # Add the key value pair to the dictionary
            key = ast.literal_eval(key)
            value = ast.literal_eval(value)
            my_dict[key] = value
    return my_dict


def closest_point_on_cylinder(point, height, rad, colorDictP):
    # point is at arbitrary x, y, and z
    x = point[0]/math.sqrt(point[0]**2 + point[1]**2)
    y = point[1]/math.sqrt(point[0]**2 + point[1]**2)
    x = x * rad
    y = y * rad

    point_color = colorDictP[tuple(point)]

    if point[2] >= (height):  # point lies above cylinder model in z
        if color_match(point_color, (240/360, 1.0, 1.0)):
            z = height
        else:
            z = 50
    elif point[2] <= 0:  # point lies below cylinder model in z
        if color_match(point_color, (0.0, 1.0, 1.0)):
            z = 0
        else:
            z = height-50
    # point lies somewhere within blue range of cylinder model
    elif (height-50) <= point[2] < height:
        if color_match(point_color, (240/360, 1.0, 1.0)):
            z = point[2]
        else:
            z = 50
    else:  # point lies somewhere within red range of cylinder model
        if color_match(point_color, (0.0, 1.0, 1.0)):
            z = point[2]
        else:
            z = height-50
    return [x, y, z]


# matches each of the points in one array to its closest corresponding point in the other array
def match(point_cloud_p, matchDict, it, q_centroid, colorDictP, modelBlueRange, modelRedRange):
    if it > 0:
        matchDict.clear()
    for i, point_p in enumerate(point_cloud_p):
        point_q = tuple(closest_point_on_cylinder(
            point_p, 300, 21.9/2, colorDictP))
        point_p = tuple(point_p)
        matchDict[point_p] = point_q


def error(point_cloud_p, point_cloud_q, b, quat, matchDict, colorDictP, modelBlueRange, modelRedRange):
    tot = 0
    quat_star = quat_conjugate(quat)
    for point_p in point_cloud_p:
        point_p = tuple(point_p)
        # point_q = matchDict[point_p]
        point_q = closest_point_on_cylinder(
            point_p, 300, 21.9/2, colorDictP)
        Rp_i_left = quat_mult(quat, point_p)
        Rp_i_right = quat_mult(Rp_i_left, quat_star)
        Rp_i = Rp_i_right
        Rp_i = extract_vect_from_quat(Rp_i)
        # curr = Rp_i + b - point_q
        curr = [Rp_i[0] + b[0] - point_q[0], Rp_i[1] +
                b[1] - point_q[1], Rp_i[2] + b[2] - point_q[2]]
        norm_squared = (math.sqrt(curr[0]**2 + curr[1]**2 + curr[2]**2))**2
        tot += norm_squared
        color_p = colorDictP[point_p]
    return tot


def add_noise(point_cloud_p, colorDictP):
    for i, point_p in enumerate(point_cloud_p):
        point_color = colorDictP[tuple(point_p)]
        colorDictP[tuple(point_p)] = ()
        noise_x = random.uniform(0, .1)
        noise_y = random.uniform(0, .1)
        noise_z = random.uniform(0, .1)
        point_p_x = point_p[0] + noise_x
        point_p_y = point_p[1] + noise_y
        point_p_z = point_p[2] + noise_z
        point_cloud_p[i][0] = point_p_x
        point_cloud_p[i][1] = point_p_y
        point_cloud_p[i][2] = point_p_z
        point_p = [point_p_x, point_p_y, point_p_z]
        colorDictP[tuple(point_p)] = point_color
    return point_cloud_p


def is_negative():
    rand_int = random.randint(0, 1)
    rand_sign = -1 if rand_int == 0 else 1
    return rand_sign


# cylinder point cloud: r = radius; h = height
def generate_point_cloud_p(r, height, colorDict):
    # generate 100 points on the cylinder
    points = []
    num_points = 100

    for i in range(num_points):
        point = []
        x = random.uniform(-21.9/2, 21.9/2)
        sign_y = is_negative()
        y = (math.sqrt(r**2-x**2)) * sign_y
        if random.choice([True, False]):
            z = random.uniform(0, 50)
        else:
            z = random.uniform(height-50, height)
        point = [x, y, z]
        points.append(point)
    for point in points:
        if point[2] >= height-50:
            # blue hsv values
            hue = 240/360
            s = 1.0
            v = 1.0
            point_tup = tuple(point)
            colorDict[point_tup] = (hue, s, v)
        elif point[2] <= 50:
            # red hsv values
            # values below are for purposes of plotting using ax.scatter
            hue = 0.0
            s = 1.0
            v = 1.0
            point_tup = tuple(point)
            colorDict[point_tup] = (hue, s, v)
        else:
            # wood hsv values --- should never be accessed
            hue = 17/360
            s = 125/255
            v = 210/255
            point_tup = tuple(point)
            colorDict[point_tup] = (hue, s, v)
    return points, colorDict


# return 2 lists of points: red and blue. Uses distance between points.
def eliminate_outliers_lists_dist_pts(d):

    red_points = []
    blue_points = []

    for key in d:
        curr = d[key]
        if curr[0] == 0.0:  # red
            red_points.append(key)
        else:  # blue
            blue_points.append(key)

    # farthest possible distance between two points in the same bounding box
    max_dist_in_mm = 54.586

    farthest_red_pts = farthest_points_in_color_region(red_points)
    farthest_red_dist = eucl_dist(farthest_red_pts[0], farthest_red_pts[1])
    farthest_blue_pts = farthest_points_in_color_region(blue_points)
    farthest_blue_dist = eucl_dist(farthest_blue_pts[0], farthest_blue_pts[1])

    # print("Farthest red points: ", farthest_red_pts)
    # print("Distance between farthest red points: ",
    #       farthest_red_dist)
    # print("Num red points: ", len(red_points))

    while farthest_red_dist > max_dist_in_mm:
        del d[farthest_red_pts[0]]
        red_points.remove(farthest_red_pts[0])
        farthest_red_pts = farthest_points_in_color_region(red_points)
        farthest_red_dist = eucl_dist(farthest_red_pts[0], farthest_red_pts[1])

    # print("Farthest red points: ", farthest_red_pts)
    # print("Distance between farthest red points: ",
    #       farthest_red_dist)
    # print("Num red points: ", len(red_points))

    # print("Farthest blue points: ", farthest_blue_pts)
    # print("Distance between farthest blue points: ",
    #       farthest_blue_dist)
    # print("Num blue points: ", len(blue_points))

    while farthest_blue_dist > max_dist_in_mm:
        del d[farthest_blue_pts[0]]
        blue_points.remove(farthest_blue_pts[0])
        farthest_blue_pts = farthest_points_in_color_region(blue_points)
        farthest_blue_dist = eucl_dist(
            farthest_blue_pts[0], farthest_blue_pts[1])

    # print("Farthest blue points: ", farthest_blue_pts)
    # print("Distance between farthest blue points: ",
    #       farthest_blue_dist)
    # print("Num blue points: ", len(blue_points))

    return red_points, blue_points


# return 1 dict of points: red and blue. Uses distance between points.
def eliminate_outliers_dict_dist_pts(d):
    red_points = []
    blue_points = []

    for key in d:
        curr = d[key]
        if curr[0] == 0.0:  # red
            red_points.append(key)
        else:  # blue
            blue_points.append(key)

    # farthest possible distance between two points in the same bounding box
    max_dist_in_mm = 54.586

    farthest_red_pts = farthest_points_in_color_region(red_points)
    farthest_red_dist = eucl_dist(farthest_red_pts[0], farthest_red_pts[1])
    farthest_blue_pts = farthest_points_in_color_region(blue_points)
    farthest_blue_dist = eucl_dist(farthest_blue_pts[0], farthest_blue_pts[1])

    while farthest_red_dist > max_dist_in_mm:
        del d[farthest_red_pts[0]]
        red_points.remove(farthest_red_pts[0])
        farthest_red_pts = farthest_points_in_color_region(red_points)
        farthest_red_dist = eucl_dist(farthest_red_pts[0], farthest_red_pts[1])

    while farthest_blue_dist > max_dist_in_mm:
        del d[farthest_blue_pts[0]]
        blue_points.remove(farthest_blue_pts[0])
        farthest_blue_pts = farthest_points_in_color_region(blue_points)
        farthest_blue_dist = eucl_dist(
            farthest_blue_pts[0], farthest_blue_pts[1])

    retval = {}
    for point in red_points:
        retval[point] = (0.0, 1.0, 1.0)
    for point in blue_points:
        retval[point] = (0.66, 1.0, 1.0)
    return retval


def plot(point_cloud_p, colorDict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # extract x, y, z coordinates from arr1 and arr2
    point_cloud_p_x = []
    point_cloud_p_y = []
    point_cloud_p_z = []
    point_color_arr = []
    for point_p in point_cloud_p:
        point_cloud_p_x.append(point_p[0])
        point_cloud_p_y.append(point_p[1])
        point_cloud_p_z.append(point_p[2])
        point_color_arr.append(colorDict[tuple(point_p)])

    # Define the cylinder height and radius
    h = 300  # mm; approx. equivalent to 12 inches
    r = 21.9/2  # mm; measured radius from diameter. Approx. equivalent to 0.435in radius or 0.87in diameter

    # Define the number of points to use for the cylinder surface
    num_points = 1200

    # Define the colors for each half of the cylinder
    colorWood = (17/360, 125/255, 210/255)  # HSV values for wood
    color1 = (240/360, 1.0, 1.0)  # HSV values for blue
    color2 = (0.0, 1.0, 1.0)  # HSV values for red

    # Convert HSV colors to RGB colors
    colorWood_rgb = colorsys.hsv_to_rgb(
        colorWood[0], colorWood[1], colorWood[2])
    color1_rgb = colorsys.hsv_to_rgb(color1[0], color1[1], color1[2])
    color2_rgb = colorsys.hsv_to_rgb(color2[0], color2[1], color2[2])

    # Create the cylinder surface
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.linspace(0, h, num_points)
    r, theta = np.meshgrid(r, theta)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z, _ = np.meshgrid(z, theta)

    bottom_height = 50  # mm
    top_height = h-50  # mm

    # Define the color values for each section of the cylinder surface
    facecolors = np.zeros((num_points, num_points, 4))
    for i in range(num_points):
        if i < num_points * bottom_height / 300:
            facecolors[:, i, :] = (*color2_rgb, 1)
        elif i < num_points * top_height / 300:
            facecolors[:, i, :] = (*colorWood_rgb, 1)
        else:
            facecolors[:, i, :] = (*color1_rgb, 1)

    # plot the points
    # ax.scatter(point_cloud_p_x, point_cloud_p_y, point_cloud_p_z, c='r', marker='o')
    for i in range(len(point_cloud_p_x)):
        point_color = point_color_arr[i]
        hue = point_color[0]
        saturation = point_color[1]
        value = point_color[2]
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        color = np.array([r, g, b])
        color = color.reshape(1, -1)
        ax.scatter(point_cloud_p_x[i], point_cloud_p_y[i],
                   point_cloud_p_z[i], c=color, marker='o')

    # Plot the cylinder surface
    ax.plot_surface(x, y, z, facecolors=facecolors, alpha=0.3, shade=True)

    # Set the axis limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

# end ICP helper functions

# Below are all the Kinect helper functions

### start Kinect function definitions###

# find the two largest rectangles and give the length from one end of one to the other


def findLargestCurrRect(rect_dict):
    max_area = -1
    # initializes largest_rect to be the first rect in the keys array of rect_dict
    largest_rect = next(iter(rect_dict))
    for rect in rect_dict.keys():
        dims = rect[1]  # width and height of curr rect
        w = dims[0]
        h = dims[1]
        area = w * h
        if area > max_area:
            max_area = area
            largest_rect = rect

    rect = largest_rect
    box = rect_dict[rect]
    return rect, box

# returns the distance between two points


def distance(x1, y1, x2, y2):
    # Calculate the distance between the two points
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Return the distance
    return dist

# find the endpoints of the line to be drawn. In theory, these will define the 2D boundary of the rod.


def findEndPointsLine(wood_box, color_box):
    # midpoint of top-left and bottom-left of rectangle
    midpoint_left_wood_box = (wood_box[0]+wood_box[3])//2
    # midpoint of top-right and bottom-right of rectangle
    midpoint_right_wood_box = (wood_box[1]+wood_box[2])//2
    # midpoint of top-left and bottom-left of rectangle
    midpoint_left_color_box = (color_box[0]+color_box[3])//2
    # midpoint of top-right and bottom-right of rectangle
    midpoint_right_color_box = (color_box[1]+color_box[2])//2
    dist_lw_ro = distance(midpoint_left_wood_box[0], midpoint_left_wood_box[1],
                          midpoint_right_color_box[0], midpoint_right_color_box[1])
    dist_lo_rw = distance(midpoint_left_color_box[0], midpoint_left_color_box[1],
                          midpoint_right_wood_box[0], midpoint_right_wood_box[1])
    endpoints = []
    if (dist_lw_ro > dist_lo_rw):  # wood is on the left
        endpoints = [midpoint_left_wood_box, midpoint_right_color_box]
    else:
        endpoints = [midpoint_left_color_box, midpoint_right_wood_box]
    return endpoints


def findEndPointsLineAndCorners(wood_box, color_box):
    # midpoint of top-left and bottom-left of rectangle
    midpoint_left_wood_box = (wood_box[0]+wood_box[3])//2
    # midpoint of top-right and bottom-right of rectangle
    midpoint_right_wood_box = (wood_box[1]+wood_box[2])//2
    # midpoint of top-left and bottom-left of rectangle
    midpoint_left_color_box = (color_box[0]+color_box[3])//2
    # midpoint of top-right and bottom-right of rectangle
    midpoint_right_color_box = (color_box[1]+color_box[2])//2
    dist_lw_ro = distance(midpoint_left_wood_box[0], midpoint_left_wood_box[1],
                          midpoint_right_color_box[0], midpoint_right_color_box[1])
    dist_lo_rw = distance(midpoint_left_color_box[0], midpoint_left_color_box[1],
                          midpoint_right_wood_box[0], midpoint_right_wood_box[1])
    endpoints = []
    corners = []
    if (dist_lw_ro > dist_lo_rw):  # wood is on the left
        endpoints = [midpoint_left_wood_box, midpoint_right_color_box]
        corners = [wood_box[0], wood_box[3], color_box[1], color_box[2]]
    else:
        endpoints = [midpoint_left_color_box, midpoint_right_wood_box]
        corners = [color_box[0], color_box[3], wood_box[1], wood_box[2]]
    return endpoints, corners


def is_negative():
    rand_int = random.randint(0, 1)
    rand_sign = -1 if rand_int == 0 else 1
    return rand_sign

# returns the minimum distance rows from between two rows of a bounding box


def minDistRows(boxPoints):
    minDist = float('inf')
    point1row = boxPoints[0][0]
    point2row = boxPoints[1][0]
    point3row = boxPoints[2][0]
    point4row = boxPoints[3][0]
    rows = [point1row, point2row, point3row, point4row]
    row1idx = 0
    row2idx = 0
    for i in range(0, len(rows)-1):
        for j in range(i+1, len(rows)):
            if rows[i] == rows[j]:
                continue
            dist = abs(rows[i] - rows[j])
            if dist < minDist:
                minDist = dist
                row1idx = i
                row2idx = j
    return [rows[row1idx], rows[row2idx]]

# returns the minimum distance columns from between two columns of a bounding box


def minDistCols(boxPoints):
    minDist = float('inf')
    point1col = boxPoints[0][1]
    point2col = boxPoints[1][1]
    point3col = boxPoints[2][1]
    point4col = boxPoints[3][1]
    cols = [point1col, point2col, point3col, point4col]
    col1idx = 0
    col2idx = 0
    for i in range(0, len(cols)-1):
        for j in range(i+1, len(cols)):
            if cols[i] == cols[j]:
                continue
            dist = abs(cols[i] - cols[j])
            if dist < minDist:
                minDist = dist
                col1idx = i
                col2idx = j
    return [cols[col1idx], cols[col2idx]]

# calculates the distance between two points


def eucl_dist(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance


def separate_into_color_regions(d):  # return 2 lists of points: red and blue
    red_points = []
    blue_points = []

    for key in d:
        curr = d[key]
        if curr[0] == 0.0:  # red
            red_points.append(key)
        else:  # blue
            blue_points.append(key)
    return red_points, blue_points

# return 1 dict of points: red and blue. Uses distance from centroid.


def eliminate_outliers_dict_dist_centroid(d):
    red_points, blue_points = separate_into_color_regions(d)
    # farthest possible distance between two points from the centroid of the color region
    max_dist_in_mm = 27.29

    # find centroid of red points
    red_centroid = point_cloud_centroid(red_points)
    # find centroid of blue points
    blue_centroid = point_cloud_centroid(blue_points)

    for point in red_points:
        if eucl_dist(point, red_centroid) > max_dist_in_mm:
            del d[point]

    for point in blue_points:
        if eucl_dist(point, blue_centroid) > max_dist_in_mm:
            del d[point]

    return d

# finds the two points in the color region that are farthest apart and stores them in an array


def farthest_points_in_color_region(color_pts_arr):
    max_dist = float('-inf')
    max_dist_pts = []
    for i in range(len(color_pts_arr)):
        for j in range(i+1, len(color_pts_arr)):
            dist = eucl_dist(color_pts_arr[i], color_pts_arr[j])
            if dist > max_dist:
                max_dist = dist
                max_dist_pts = [color_pts_arr[i], color_pts_arr[j]]
    return max_dist_pts


def count_red_pts(d):
    count = 0
    for key in d:
        curr = d[key]
        if curr[0] == 0.0:
            count += 1
    return count


def count_blue_pts(d):
    count = 0
    for key in d:
        curr = d[key]
        if curr[0] == (240.0 / 360.0):
            count += 1
    return count

### end Kinect function definitions###


def eliminate_nans():
    with open('exportDict0.txt', 'r') as f_in, open('exportDict0new.txt', 'w') as f_out:
        for line in f_in:
            if 'nan' not in line:
                f_out.write(line)
