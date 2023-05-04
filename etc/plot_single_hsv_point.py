import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define the HSV values for the color
hsv_pixel = (105, 255, 187)
h = hsv_pixel[0]
s = hsv_pixel[1]
v = hsv_pixel[2]


# Convert the HSV values to RGB values
hsv_pixel = np.array([h, s, v], dtype=np.uint8)  # replace with your values
rgb_pixel = cv2.cvtColor(np.array([[hsv_pixel]]).astype(np.float32), cv2.COLOR_HSV2RGB)[0, 0]

# Plot a color swatch using the RGB values
plt.plot([0, 1], [0, 0], color=rgb_pixel, linewidth=10)
plt.show()