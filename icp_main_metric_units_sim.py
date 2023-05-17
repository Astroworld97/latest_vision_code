# quaternion implementation from 577 class notes. All units in inches.
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from numpy.linalg import eig
from helpers import *
import colorsys
import hashlib
import copy

# main
# section 1: define constants and data structures
maxIterations = 10000
tolerance = .1
matchDict = {}
colorDictP = {}
# Define the cylinder height and radius
h = 300  # mm; approximately equivalent to 12 inches (12 in = 304.8 mm)
# mm; Measured radius from diameter. Approximately equivalent to measured 0.435in radius or 0.87in diameter.
r = 21.9/2
# section of analytical model with blue color tape; approximately equivalent to 2 inches (50.8 mm = 2 inches)
modelBlueRange = [h-50, h]
# section of analytical model that is red color tape; approximately equivalent to 2 inches (50.8 mm = 2 inches)
modelRedRange = [0.00, 50]
# section 2: define arrays (aka point clouds) and initialize dictionaries
point_cloud_q, colorDictP = generate_point_cloud_p(r, h, colorDictP)
point_cloud_p = point_cloud_q.copy()
plot(point_cloud_p, colorDictP)
point_cloud_p, colorDictP = apply_initial_translation_and_rotation(
    point_cloud_p, colorDictP)
point_cloud_p = add_noise(point_cloud_p, colorDictP)
plot(point_cloud_p, colorDictP)
M = np.zeros((4, 4))
b = 0
quat = [0, 0, 0, 0]  # aka quat
p_centroid = point_cloud_centroid(point_cloud_p)
q_centroid = point_cloud_centroid(point_cloud_q)
# q_centroid = [0,0,300/2]

# #section 3: iterate (rotation)
point_cloud_p_best = point_cloud_p
best_err = 10000000
for i in range(maxIterations):

    match(point_cloud_p, matchDict, i, q_centroid, colorDictP, modelBlueRange,
          modelRedRange)  # fill the matchDict with the current matches

    if (i > 0):  # only check for error after the 0th loop
        point_cloud_q = []
        for i, point_p in enumerate(point_cloud_p):
            point_q = closest_point_on_cylinder(
                point_p, h, r, colorDictP)
            point_cloud_q.append(point_q)
        err = error(point_cloud_p, point_cloud_q, b, quat,
                    matchDict, colorDictP, modelBlueRange, modelRedRange)
        if err < best_err:
            best_err = err
            print("update")
            point_cloud_p_best = point_cloud_p
            colorDictP_best = colorDictP
        print(err)
        if err < tolerance:
            break

    p_centroid = point_cloud_centroid(point_cloud_p)
    M = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    for point_p in point_cloud_p:
        point_p = tuple(point_p)
        point_q = matchDict[point_p]
        p_prime = calc_single_prime(point_p, p_centroid)
        q_prime = calc_single_prime(point_q, q_centroid)
        P_i = create_prime_matrix_p(p_prime)
        Q_i = create_prime_matrix_q(q_prime)
        M_i = calc_single_M(P_i, Q_i)
        toFill = [[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]
        for i in range(len(M_i)):
            for j in range(len(M_i[0])):
                toFill[i][j] = M_i[i][j] + M[i][j]
        M = toFill
    # plot(point_cloud_p, colorDictP)

    quat = calc_quat(M)  # aka quat
    norm = quat_norm(quat)
    quat = [quat[0]/norm, quat[1]/norm, quat[2]/norm, quat[3]/norm]
    quat_star = quat_conjugate(quat)  # conjugate of q, aka quat
    b = calc_b(q_centroid, p_centroid, quat, quat_star)
    print("b: " + str(b))
    for i, point_p in enumerate(point_cloud_p):
        left_curr = quat_mult(quat, point_p)
        right_curr = quat_mult(left_curr, quat_star)
        Rp = [right_curr[1], right_curr[2], right_curr[3]]
        point_color = colorDictP[tuple(point_p)]
        colorDictP[tuple(point_p)] = ()
        point_p = [Rp[0] + b[0], Rp[1] + b[1], Rp[2] + b[2]]
        colorDictP[tuple(point_p)] = point_color
        point_cloud_p[i] = point_p
print(best_err)
plot(point_cloud_p_best, colorDictP_best)

err = error(point_cloud_p, point_cloud_q, b, quat, matchDict,
            colorDictP_best, modelBlueRange, modelRedRange)
print(p_centroid)
print(q_centroid)
# plot(point_cloud_p, colorDict)
