import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from helpers_kinect import *
import random
import matplotlib.pyplot as plt
import colorsys
import json

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
lower1 = np.array([83, 166, 173])
upper1 = np.array([113, 196, 233])
### red paper###
lower2 = np.array([160, 160, 90])
upper2 = np.array([190, 200, 250])

# angle_arr, which will contain all the angles over the span of the camera being open
angle_arr = []

count = 0
while count < 30:
    frames = listener.waitForNewFrame()
    color = frames["color"]
    depth = frames["depth"]
    color_data = color.asarray()
    color_data = color_data[:, :, :3]
    depth_data = depth.asarray(np.uint8)
    # extract the last channel which contains the depth information
    depth_channel = depth_data[:, :, -1]
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
    largest_red_rect, largest_red_box = findLargestCurrRect(
        rect_dict_red)

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    cv2.drawContours(frame, [largest_blue_box], 0, (255, 0, 0), 2)
    cv2.drawContours(frame, [largest_red_box], 0, (0, 255, 0), 2)

    # Define the colors for each half of the cylinder

    color1 = (0.66, 1.0, 1.0)  # HSV values for blue
    color2 = (0.0, 1.0, 1.0)  # HSV values for red

    # plot and store blue points along blue bounding box border
    for i in range(10):
        x0 = np.linspace(largest_blue_box[0][0], largest_blue_box[1][0], 10)[i]
        y0 = np.linspace(largest_blue_box[0][1], largest_blue_box[1][1], 10)[i]
        z = depth_channel[int(y0), int(x0)]
        colorDict[(x0, y0, z)] = color1
        cv2.circle(frame, (int(x0), int(y0)), 2,
                   (255, 0, 0), -1)  # BGR, not RGB
        x1 = np.linspace(largest_blue_box[1][0], largest_blue_box[2][0], 10)[i]
        y1 = np.linspace(largest_blue_box[1][1], largest_blue_box[2][1], 10)[i]
        z = depth_channel[int(y1), int(x1)]
        colorDict[(x1, y1, z)] = color1
        cv2.circle(frame, (int(x1), int(y1)), 2, (255, 0, 0), -1)
        x2 = np.linspace(largest_blue_box[2][0], largest_blue_box[3][0], 10)[i]
        y2 = np.linspace(largest_blue_box[2][1], largest_blue_box[3][1], 10)[i]
        z = depth_channel[int(y2), int(x2)]
        colorDict[(x2, y2, z)] = color1
        cv2.circle(frame, (int(x2), int(y2)), 2, (255, 0, 0), -1)
        x3 = np.linspace(largest_blue_box[3][0], largest_blue_box[0][0], 10)[i]
        y3 = np.linspace(largest_blue_box[3][1], largest_blue_box[0][1], 10)[i]
        z = depth_channel[int(y3), int(x3)]
        colorDict[(x3, y3, z)] = color1
        cv2.circle(frame, (int(x3), int(y3)), 2, (255, 0, 0), -1)

    # plot and store blue points inside blue bounding box border
    for i in range(20):
        min_row = min(point[0] for point in largest_blue_box)
        max_row = max(point[0] for point in largest_blue_box)
        min_col = min(point[1] for point in largest_blue_box)
        max_col = max(point[1] for point in largest_blue_box)
        r = random.randint(int(min_row), int(max_row))
        c = random.randint(int(min_col), int(max_col))
        z = depth_channel[int(r), int(c)]
        toAdd = (r, c, z)
        colorDict[toAdd] = color1
        cv2.circle(frame, (int(r), int(c)), 2, (255, 0, 0), -1)

    # plot and store red points along red bounding box border
    for i in range(10):
        x0 = np.linspace(largest_red_box[0]
                         [0], largest_red_box[1][0], 10)[i]
        y0 = np.linspace(largest_red_box[0]
                         [1], largest_red_box[1][1], 10)[i]
        z = depth_channel[int(y0), int(x0)]
        colorDict[(x0, y0, z)] = color2
        cv2.circle(frame, (int(x0), int(y0)), 2, (0, 0, 255), -1)
        x1 = np.linspace(largest_red_box[1]
                         [0], largest_red_box[2][0], 10)[i]
        y1 = np.linspace(largest_red_box[1]
                         [1], largest_red_box[2][1], 10)[i]
        z = depth_channel[int(y1), int(x1)]
        colorDict[(x1, y1, z)] = color2
        cv2.circle(frame, (int(x1), int(y1)), 2, (0, 0, 255), -1)
        x2 = np.linspace(largest_red_box[2]
                         [0], largest_red_box[3][0], 10)[i]
        y2 = np.linspace(largest_red_box[2]
                         [1], largest_red_box[3][1], 10)[i]
        z = depth_channel[int(y2), int(x2)]
        colorDict[(x2, y2, z)] = color2
        cv2.circle(frame, (int(x2), int(y2)), 2, (0, 0, 255), -1)
        x3 = np.linspace(largest_red_box[3]
                         [0], largest_red_box[0][0], 10)[i]
        y3 = np.linspace(largest_red_box[3]
                         [1], largest_red_box[0][1], 10)[i]
        z = depth_channel[int(y3), int(x3)]
        colorDict[(x3, y3, z)] = color2
        cv2.circle(frame, (int(x3), int(y3)), 2, (0, 0, 255), -1)

    # plot and store red points inside red bounding box border
    for i in range(20):
        min_row = min(point[0] for point in largest_red_box)
        max_row = max(point[0] for point in largest_red_box)
        min_col = min(point[1] for point in largest_red_box)
        max_col = max(point[1] for point in largest_red_box)
        r = random.randint(int(min_row), int(max_row))
        c = random.randint(int(min_col), int(max_col))
        z = depth_channel[int(r), int(c)]
        toAdd = (r, c, z)
        colorDict[toAdd] = color2
        cv2.circle(frame, (int(r), int(c)), 2, (0, 0, 255), -1)

    endpoints, corners = findEndPointsLineAndCorners(
        largest_blue_box, largest_red_box)
    # cv2.line(frame, endpoints[0], endpoints[1], (37, 65, 23), 2)
    cv2.imwrite('video_test' + str(count) + '.jpg', frame)
    # out.write(frame)
    # Exit on 'q' press
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # Note to self: remember images are indexed as rows and columns. Use r and c to iterate, rather than x and y.
    ### y = rows = r; height###
    ### x = columns = c; width###

    # reconstruct original image as plot
    # Create an empty plot for xy pixel coordinates and z depth values in mm
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    keys = list(colorDict.keys())
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
        z = point[2]
        # ax1.scatter(col, 424-row, z, c=color, marker='o')
        x, y, z = registration.getPointXYZ(undistorted, row, col)
        x = x*39.3701  # convert meters to inches
        y = y*39.3701  # convert meters to inches
        z = z*39.3701  # convert meters to inches
        new_key = (x, y, z)
        exportDict[new_key] = hsv
        ax2.scatter(x, y, z, c=color, marker='o')
    # Set the axis limits and labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # save the plot as a pdf
    # fig1.savefig('point_cloud_' + str(count) + '_in_pixels.pdf')
    # fig1.savefig('point_cloud_' + str(count) + '_in_pixels.png')
    fig2.savefig('point_cloud_' + str(count) + '_in_inches.pdf')
    fig2.savefig('point_cloud_' + str(count) + '_in_inches.png')

    colorDict_list.append(colorDict)

    # Open a file for writing
    with open('exportDict ' + str(count) + '.txt', 'w') as f:
        for key, value in exportDict.items():
            f.write('{}:{}\n'.format(key, value))

    colorDict.clear()
    # plt.clf()

    # Show the plot
    fig2.show()
    plt.show()
    listener.release(frames)

    print(count)
    count += 1

# cv2.waitKey(0)
cv2.destroyAllWindows()
