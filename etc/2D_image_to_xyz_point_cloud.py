import cv2
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import random

###Note to self: remember images are indexed as rows and columns. Use r and c to iterate, rather than x and y.
###y = rows = r###
###x = columns = c###

def convert_from_opencv_hsv_to_regular_hsv(opencv_hsv):
    regular_hsv = (opencv_hsv[0]*360/180), (opencv_hsv[1]*100/255), (opencv_hsv[2]*100/255)
    return regular_hsv

# Load image
image = cv2.imread('flag.png')
height, width, channels = image.shape
print(height)
print(width)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create point cloud
point_cloud = []
colorDict = {}
for r in range(height):
    for c in range(width):
        
        # Get HSV of pixel
        hsv = tuple(hsv_image[r, c])
        # z = random.uniform(-0.435, 0.435) #depth of point cloud, set to a random number between -.435 and .435
        z = 0
        pixel = tuple([c, r, z]) #technically, this is the same thing as [x,y,z]
        colorDict[pixel] = hsv

#reconstruct original image as plot
# Create an empty plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
keys = list(colorDict.keys())
print(len(keys))
# Loop through each pixel and set its color using the given HSV values
counter = 0  # initialize counter variable
for point in keys:
    counter += 1  # increment counter on each iteration
    if counter % 10000 == 0:  # check if counter is divisible by 100
        # Get the HSV values for the current pixel
        hsv = colorDict[tuple(point)]
        h = hsv[0]/180
        s = hsv[1]/255
        v = hsv[2]/255
        
        # Convert the HSV values to RGB values
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = np.array([r,g,b])
        color=color.reshape(1,-1)

        # Set the color of the pixel (aka point) in the image using the RGB values
        col = point[0]
        row = point[1]
        z = point[2]
        ax.scatter(col, height-row, z, c=color, marker='o')
# Set the axis limits and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()



        
        
        
        