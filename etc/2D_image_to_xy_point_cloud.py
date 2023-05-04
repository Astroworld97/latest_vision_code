import cv2
import numpy as np
import colorsys
import matplotlib.pyplot as plt

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
        pixel = tuple([r, c])
        colorDict[pixel] = hsv

#print first pixel's coords and color in hsv
first_item = next(iter(colorDict.items()))
print(first_item[0])  # prints key
print(first_item[1])  # prints value
hsv1 = first_item[1]
hsv2 = ()
for k, v in colorDict.items():
        if v != (105, 255, 187):
            print (k)
            print(v)
            hsv2 = v
            break

#reconstruct original image as plot
# Create an empty plot
fig = plt.figure()
ax = fig.add_subplot(111)
# Loop through each pixel and set its color using the given HSV values
for row in range(0, height, 100):
    for col in range(0, width, 100):
        # Get the HSV values for the current pixel
        hsv = colorDict[(row,col)]
        h = hsv[0]/180
        s = hsv[1]/255
        v = hsv[2]/255
        
        # Convert the HSV values to RGB values
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        color = np.array([r,g,b])
        color=color.reshape(1,-1)

        # Set the color of the pixel (aka point) in the image using the RGB values
        ax.scatter(col, height-row, c=color, marker='o')
# Set the axis limits and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Show the plot
plt.show()



        
        
        
        