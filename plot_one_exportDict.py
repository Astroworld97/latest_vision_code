import numpy as np
from helpers_kinect import *
import matplotlib.pyplot as plt
import colorsys
from icp_helpers_metric_units import *

# Create an empty plot for xy pixel coordinates and z depth values in mm
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
colorDict = txt_to_dict('exportDict0.txt')
keys = list(colorDict.keys())
# Loop through each pixel and set its color using the given HSV values
exportDict = {}
for point in keys:
    # Get the HSV values for the current pixel
    hsv = colorDict[tuple(point)]
    h = hsv[0]
    s = hsv[1]
    v = hsv[2]

    # Convert the HSV values to RGB values
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    color = np.array([r, g, b])
    color = color.reshape(1, -1)

    # Set the color of the pixel (aka point) in the image using the RGB values
    x = point[0]
    y = point[1]
    z = point[2]
    new_key = (x, -y, z)
    ax1.scatter(x, -y, z, c=color, marker='o')

# Set the axis limits and labels
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

plt.show()
