import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
# from helpers_kinect import *
from collections import defaultdict
from helpers import *
import random
import matplotlib.pyplot as plt
import colorsys
import math
from numpy import nan
### start Kinect initialization###
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()  # kinect object
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()
### end Kinect initialization###

# Set up video writer --> might be needed to save the video
# dims_frame = cv2.imread('registered_frame.jpg', cv2.IMREAD_COLOR)
# height, width, channels = dims_frame.shape
fps = 30.0  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (512, 424))

# angle_arr, which will contain all the angles over the span of the camera being open
angle_arr = []

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

color_data = None
depth_data = None

# Define the dictionary
colorDict = {}
colorDict_list = []
# Define the range of the color to be segmented
### blue paper###
lower1 = np.array([83, 166, 0])
upper1 = np.array([113, 196, 255])

### red paper###
lower2 = np.array([160, 160, 0])
upper2 = np.array([190, 200, 255])

# angle_arr, which will contain all the angles over the span of the camera being open
angle_arr = []

count = 0
while count < 1:
    frames = listener.waitForNewFrame()
    color = frames["color"]
    depth = frames["depth"]
    registration.apply(color, depth, undistorted, registered)
    registered_data = registered.asarray(np.uint8)
    registered_data = registered_data[:, :, :3]
    frame = registered_data

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks based on the defined range
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # Apply the mask to the original image
    result1 = cv2.bitwise_and(frame, frame, mask=mask1)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)

    # Find contours of the segmented object
    contoursBlue, _ = cv2.findContours(
        mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursRed, _ = cv2.findContours(
        mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contoursBlue) == 0:
        print("No contours found for blue segment.")
        listener.release(frames)
        continue

    if len(contoursRed) == 0:
        print("No contours found for red segment.")
        listener.release(frames)
        continue

    # declare rectangle dictionary (keys: rects, values: boxes)
    rect_dict_blue = defaultdict(int)
    rect_dict_red = defaultdict(int)

    # Find the bounding boxes of the object
    for cnt in contoursBlue:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_dict_blue[rect] = box

    for cnt in contoursRed:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_dict_red[rect] = box

    # Find the two rectangles with the largest area (one is blue paper color, the other is red paper color)
    largest_blue_rect, largest_blue_box = findLargestCurrRect(rect_dict_blue)
    largest_red_rect, largest_red_box = findLargestCurrRect(rect_dict_red)

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    cv2.drawContours(frame, [largest_blue_box], 0, (255, 0, 0), 2)
    cv2.drawContours(frame, [largest_red_box], 0, (0, 255, 0), 2)

    # Define the colors for each half of the cylinder

    color1 = (0.66, 1.0, 1.0)  # HSV values for blue
    color2 = (0.0, 1.0, 1.0)  # HSV values for red

    # plot and store blue points inside blue bounding box border
    for i in range(60):
        min_x = min(point[0] for point in largest_blue_box)
        max_x = max(point[0] for point in largest_blue_box)
        min_y = min(point[1] for point in largest_blue_box)
        max_y = max(point[1] for point in largest_blue_box)
        x = random.randint(int(min_x), int(max_x))
        y = random.randint(int(min_y), int(max_y))
        toAdd = (x, y)
        colorDict[toAdd] = color1
        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

    # plot and store red points inside red bounding box border
    for i in range(60):
        min_x = min(point[0] for point in largest_red_box)
        max_x = max(point[0] for point in largest_red_box)
        min_y = min(point[1] for point in largest_red_box)
        max_y = max(point[1] for point in largest_red_box)
        x = random.randint(int(min_x), int(max_x))
        y = random.randint(int(min_y), int(max_y))
        toAdd = (x, y)
        colorDict[toAdd] = color2
        cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

    cv2.imwrite('video_test' + str(count) + '.jpg', frame)
    # out.write(frame)

    # Note to self: remember images are indexed as rows and columns. Use r and c to iterate, rather than x and y.
    ### y = rows = r; height###
    ### x = columns = c; width###

    # reconstruct original image as plot
    # Create an empty plot xyz values in mm
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
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
        col = point[0]
        row = point[1]
        # z = point[2]

        x, y, z = registration.getPointXYZ(undistorted, row, col)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        new_key = (x, y, z)
        if math.isnan(point[0]):
            continue
        exportDict[new_key] = hsv
        ax2.scatter(x, y, z, c=color, marker='o')
    # Set the axis limits and labels
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # save the plot as a pdf
    fig2.savefig('point_cloud_' + str(count) + '_in_mm.pdf')
    fig2.savefig('point_cloud_' + str(count) + '_in_mm.png')

    colorDict_list.append(colorDict)

    # Open a file for writing
    with open('exportDict' + str(count) + '.txt', 'w') as f:
        for key, value in exportDict.items():
            f.write('{}:{}\n'.format(key, value))

    colorDict.clear()

    # Show the plots
    plt.show()
    listener.release(frames)

    print(count)
    count += 1

cv2.destroyAllWindows()
