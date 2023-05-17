import numpy as np
import random

largest_blue_box = [[237, 190], [250, 190], [250, 192], [237, 192]]
colorDict = {}

print(largest_blue_box[0][0])
print(largest_blue_box[0][1])
print(largest_blue_box[1][0])
print(largest_blue_box[1][1])
print(largest_blue_box[2][0])
print(largest_blue_box[2][1])
print(largest_blue_box[3][0])
print(largest_blue_box[3][1])

min_x = min(point[0] for point in largest_blue_box)
max_x = max(point[0] for point in largest_blue_box)
min_y = min(point[1] for point in largest_blue_box)
max_y = max(point[1] for point in largest_blue_box)

print(min_x)
print(max_x)
print(min_y)
print(max_y)

# color1 = (0.66, 1.0, 1.0)  # HSV values for blue
# color2 = (0.0, 1.0, 1.0)  # HSV values for red

# # plot and store blue points along blue bounding box border
# for i in range(10):
#     x0 = np.linspace(largest_blue_box[0][0], largest_blue_box[1][0], 10)[i]
#     y0 = np.linspace(largest_blue_box[0][1], largest_blue_box[1][1], 10)[i]
#     # z = depth_channel[int(y0), int(x0)]
#     # z = depth_channel[int(x0), int(y0)]
#     z = random.randint(0, 100)
#     colorDict[(x0, y0, z)] = color1
#     # colorDict[(y0, x0, z)] = color1
#     x1 = np.linspace(largest_blue_box[1][0], largest_blue_box[2][0], 10)[i]
#     y1 = np.linspace(largest_blue_box[1][1], largest_blue_box[2][1], 10)[i]
#     # z = depth_channel[int(x1), int(y1)]
#     # z = depth_channel[int(y1), int(x1)]
#     colorDict[(x1, y1, z)] = color1
#     # colorDict[(y1, x1, z)] = color1
#     x2 = np.linspace(largest_blue_box[2][0], largest_blue_box[3][0], 10)[i]
#     y2 = np.linspace(largest_blue_box[2][1], largest_blue_box[3][1], 10)[i]
#     # z = depth_channel[int(y2), int(x2)]
#     # z = depth_channel[int(x2), int(y2)]
#     z = random.randint(0, 100)
#     colorDict[(x2, y2, z)] = color1
#     # colorDict[(y2, x2, z)] = color1
#     x3 = np.linspace(largest_blue_box[3][0], largest_blue_box[0][0], 10)[i]
#     y3 = np.linspace(largest_blue_box[3][1], largest_blue_box[0][1], 10)[i]
#     # z = depth_channel[int(y3), int(x3)]
#     # z = depth_channel[int(x3), int(y3)]
#     z = random.randint(0, 100)
#     colorDict[(x3, y3, z)] = color1
#     # colorDict[(y3, x3, z)] = color1

# # plot and store blue points inside blue bounding box border
# for i in range(20):
#     min_row = min(point[0] for point in largest_blue_box)
#     max_row = max(point[0] for point in largest_blue_box)
#     min_col = min(point[1] for point in largest_blue_box)
#     max_col = max(point[1] for point in largest_blue_box)
#     r = random.randint(int(min_row), int(max_row))
#     c = random.randint(int(min_col), int(max_col))
#     # z = depth_channel[int(r), int(c)]
#     z = random.randint(0, 100)
#     toAdd = (r, c, z)
#     colorDict[toAdd] = color1
