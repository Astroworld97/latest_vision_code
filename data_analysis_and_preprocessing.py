from helpers_kinect import *
import matplotlib.pyplot as plt
import colorsys

d = txt_to_dict('exportDict0.txt')

# Uncomment to print the dictionary
# # Iterate over the key-value pairs of the dictionary
# for key, value in d.items():
#     # Format the key-value pair as a string and print it
#     print("{}: {}".format(key, value))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
keys = list(d.keys())

for point in keys:
    # Get the HSV values for the current pixel
    hsv = d[tuple(point)]
    h = hsv[0]
    s = hsv[1]
    v = hsv[2]

    # Convert the HSV values to RGB values
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    color = np.array([r, g, b])
    color = color.reshape(1, -1)
    x = point[0]
    y = point[1]
    z = point[2]
    ax1.scatter(x, y, z, c=color, marker='o')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

plt.show()

# def eliminate_outliers(d):

#     red_points = []
#     blue_points = []

#     for key in d:
#         curr = d[key]
#         if curr[0] == 0.0:  # red
#             red_points.append(key)
#         else:  # blue
#             blue_points.append(key)

#     # farthest possible distance between two points in the same bounding box
#     max_dist_in_mm = 54.586

#     farthest_red_pts = farthest_points_in_color_region(red_points)
#     farthest_red_dist = eucl_dist(farthest_red_pts[0], farthest_red_pts[1])
#     farthest_blue_pts = farthest_points_in_color_region(blue_points)
#     farthest_blue_dist = eucl_dist(farthest_blue_pts[0], farthest_blue_pts[1])

#     # print("Farthest red points: ", farthest_red_pts)
#     # print("Distance between farthest red points: ",
#     #       farthest_red_dist)
#     # print("Num red points: ", len(red_points))

#     while farthest_red_dist > max_dist_in_mm:
#         del d[farthest_red_pts[0]]
#         red_points.remove(farthest_red_pts[0])
#         farthest_red_pts = farthest_points_in_color_region(red_points)
#         farthest_red_dist = eucl_dist(farthest_red_pts[0], farthest_red_pts[1])

#     # print("Farthest red points: ", farthest_red_pts)
#     # print("Distance between farthest red points: ",
#     #       farthest_red_dist)
#     # print("Num red points: ", len(red_points))

#     # print("Farthest blue points: ", farthest_blue_pts)
#     # print("Distance between farthest blue points: ",
#     #       farthest_blue_dist)
#     # print("Num blue points: ", len(blue_points))

#     while farthest_blue_dist > max_dist_in_mm:
#         del d[farthest_blue_pts[0]]
#         blue_points.remove(farthest_blue_pts[0])
#         farthest_blue_pts = farthest_points_in_color_region(blue_points)
#         farthest_blue_dist = eucl_dist(
#             farthest_blue_pts[0], farthest_blue_pts[1])

#     # print("Farthest blue points: ", farthest_blue_pts)
#     # print("Distance between farthest blue points: ",
#     #       farthest_blue_dist)
#     # print("Num blue points: ", len(blue_points))

#     return red_points, blue_points


# # plot processed points
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')

# red_points, blue_points = eliminate_outliers(d)

# for p in red_points:
#     ax2.scatter(p[0], p[1], p[2], c='r', marker='o')

# for p in blue_points:
#     ax2.scatter(p[0], p[1], p[2], c='b', marker='o')

# # Set the axis limits and labels
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')

# plt.show()
