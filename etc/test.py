import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from collections import defaultdict
from helpers import *
import random
import matplotlib as mpl
mpl.use('tkagg')
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
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, fps, (512, 424))

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
    # rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # print("One blue point: " + str([largest_blue_box[0]]))
    # first = largest_blue_box[0][0]
    # second = largest_blue_box[0][1]
    # print(first)
    # print(second)
    # rgb = rgb_image[first,second]
    # print(rgb)
    # first = largest_blue_box[0][0]
    # second = largest_blue_box[0][1]
    # print(first)
    # print(second)
    # height, width, channels = rgb_image.shape
    # print (height, width, channels)
    # print("One blue point: " + str([largest_blue_box[0]]))
    # print("One red point: " + str(largest_red_box[0]))
    # print("largest_blue_box: " + str(largest_blue_box))
    # print("largest_red_box: " + str(largest_red_box))

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    # cv2.drawContours(frame, [largest_blue_box], 0, (255, 0, 0), 2)
    # cv2.drawContours(frame, [largest_red_box], 0, (0, 0, 255), 2)

    # Define the colors for each half of the cylinder

    color1 = (0.66, 1.0, 1.0)  # HSV values for blue
    color2 = (0.0, 1.0, 1.0)  # HSV values for red

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot and store blue points inside blue bounding box border
    for i in range(1):
        min_x = min(largest_blue_box[0][0], largest_blue_box[1][0], largest_blue_box[2][0], largest_blue_box[3][0])
        max_x = max(largest_blue_box[0][0], largest_blue_box[1][0], largest_blue_box[2][0], largest_blue_box[3][0])
        min_y = min(largest_blue_box[0][1], largest_blue_box[1][1], largest_blue_box[2][1], largest_blue_box[3][1])
        max_y = max(largest_blue_box[0][1], largest_blue_box[1][1], largest_blue_box[2][1], largest_blue_box[3][1])
        # rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # print("One blue point: " + str([largest_blue_box[0]]))
        first_blue_x = largest_blue_box[0][0]
        first_blue_y = largest_blue_box[0][1]
        print(first_blue_x)
        print(first_blue_y)
        first_red_x = largest_red_box[0][0]
        first_red_y = largest_red_box[0][1]
        print(first_red_x)
        print(first_red_y)
        x, y, z = registration.getPointXYZ(undistorted, first_blue_y, first_blue_x)
        # x, y, z = registration.getPointXYZ(undistorted, first_blue_x, first_blue_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("First blue point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='blue', marker='o')
        # ax.scatter(y, x, z, c='blue', marker='o')
        x, y, z = registration.getPointXYZ(undistorted, first_red_y, first_red_x)
        # x, y, z = registration.getPointXYZ(undistorted, first_red_x, first_red_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("First red point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='red', marker='o')
        # ax.scatter(y, x, z, c='red', marker='o')
        second_blue_x = largest_blue_box[1][0]
        second_blue_y = largest_blue_box[1][1]
        print(second_blue_x)
        print(second_blue_y)
        second_red_x = largest_red_box[1][0]
        second_red_y = largest_red_box[1][1]
        print(second_red_x)
        print(second_red_y)
        x, y, z = registration.getPointXYZ(undistorted, second_blue_y, second_blue_x)
        # x, y, z = registration.getPointXYZ(undistorted, second_blue_x, second_blue_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("Second blue point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='blue', marker='o')
        # ax.scatter(y, x, z, c='blue', marker='o')
        x, y, z = registration.getPointXYZ(undistorted, second_red_y, second_red_x)
        # x, y, z = registration.getPointXYZ(undistorted, second_red_x, second_red_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("Second red point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='red', marker='o')
        # ax.scatter(y, x, z, c='red', marker='o')
        third_blue_x = largest_blue_box[2][0]
        third_blue_y = largest_blue_box[2][1]
        print(third_blue_x)
        print(third_blue_y)
        third_red_x = largest_red_box[2][0]
        third_red_y = largest_red_box[2][1]
        print(third_red_x)
        print(third_red_y)
        x, y, z = registration.getPointXYZ(undistorted, third_blue_y, third_blue_x)
        # x, y, z = registration.getPointXYZ(undistorted, third_blue_x, third_blue_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("Third blue point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='blue', marker='o')
        # ax.scatter(y, x, z, c='blue', marker='o')
        x, y, z = registration.getPointXYZ(undistorted, third_red_y, third_red_x)
        # x, y, z = registration.getPointXYZ(undistorted, third_red_x, third_red_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("Third red point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='red', marker='o')
        # ax.scatter(y, x, z, c='red', marker='o')
        fourth_blue_x = largest_blue_box[3][0]
        fourth_blue_y = largest_blue_box[3][1]
        print(fourth_blue_x)
        print(fourth_blue_y)
        fourth_red_x = largest_red_box[3][0]
        fourth_red_y = largest_red_box[3][1]
        print(fourth_red_x)
        print(fourth_red_y)
        x, y, z = registration.getPointXYZ(undistorted, fourth_blue_y, fourth_blue_x)
        # x, y, z = registration.getPointXYZ(undistorted, fourth_blue_x, fourth_blue_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("Fourth blue point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='blue', marker='o')
        # ax.scatter(y, x, z, c='blue', marker='o')
        x, y, z = registration.getPointXYZ(undistorted, fourth_red_y, fourth_red_x)
        # x, y, z = registration.getPointXYZ(undistorted, fourth_red_x, fourth_red_y)
        x = x*1000  # convert meters to mm
        y = y*1000  # convert meters to mm
        z = z*1000  # convert meters to mm
        print("Fourth red point: " + str(x) + " , " + str(y) + " , " + str(z))
        ax.scatter(x, y, z, c='red', marker='o')
        # ax.scatter(y, x, z, c='red', marker='o')
        # centroid = [round((min_x+max_x)/2), round((min_y+max_y)/2)]
        # hustvl = hsv[centroid[1], centroid[0]]
        # print(hustvl)
        # print("Blue min x: " + str(min_x))
        # print("Blue max x: " + str(max_x))
        # print("Blue min y: " + str(min_y))
        # print("Blue max y: " + str(max_y))
        # x = random.randint(int(min_x), int(max_x))
        # y = random.randint(int(min_y), int(max_y))
        # toAdd = (x, y)
        # colorDict[toAdd] = color1
        cv2.circle(frame, (int(first_blue_x), int(first_blue_y)), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(second_blue_x), int(second_blue_y)), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(third_blue_x), int(third_blue_y)), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(fourth_blue_x), int(fourth_blue_y)), 2, (255, 0, 0), -1)
        cv2.circle(frame, (int(first_red_x), int(first_red_y)), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(second_red_x), int(second_red_y)), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(third_red_x), int(third_red_y)), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(fourth_red_x), int(fourth_red_y)), 2, (0, 0, 255), -1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect("auto", "box")
    plt.show()

    # # plot and store red points inside red bounding box border
    for i in range(1):
        min_x = min(largest_red_box[0][0], largest_red_box[1][0], largest_red_box[2][0], largest_red_box[3][0])
        max_x = max(largest_red_box[0][0], largest_red_box[1][0], largest_red_box[2][0], largest_red_box[3][0])
        min_y = min(largest_red_box[0][1], largest_red_box[1][1], largest_red_box[2][1], largest_red_box[3][1])
        max_y = max(largest_red_box[0][1], largest_red_box[1][1], largest_red_box[2][1], largest_red_box[3][1])

        # print("Red min x: " + str(min_x))
        # print("Red max x: " + str(max_x))
        # print("Red min y: " + str(min_y))
        # print("Red max y: " + str(max_y))
    #     x = random.randint(int(min_x), int(max_x))
    #     y = random.randint(int(min_y), int(max_y))
    #     toAdd = (x, y)
    #     colorDict[toAdd] = color2




    cv2.imwrite('output_image_folder/video_testing' + str(count) + '.jpg', frame)
    # out.write(frame)

    # Note to self: remember images are indexed as rows and columns. Use r and c to iterate, rather than x and y.
    ### y = rows = r; height###
    ### x = columns = c; width###

    # reconstruct original image as plot
    # Create an empty plot xyz values in mm
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, projection='3d')
    # keys = list(colorDict.keys())
    # # Loop through each pixel and set its color using the given HSV values
    # exportDict = {}
    # for point in keys:
    #     # Get the HSV values for the current pixel
    #     hsv = colorDict[tuple(point)]
    #     h = hsv[0]
    #     s = hsv[1]
    #     v = hsv[2]

    #     # Convert the HSV values to RGB values
    #     r, g, b = colorsys.hsv_to_rgb(h, s, v)
    #     color = np.array([r, g, b])
    #     color = color.reshape(1, -1)

    #     # Set the color of the pixel (aka point) in the image using the RGB values
    #     col = point[0]
    #     row = point[1]
    #     # z = point[2]

    #     x, y, z = registration.getPointXYZ(undistorted, row, col)
    #     x = x*1000  # convert meters to mm
    #     y = y*1000  # convert meters to mm
    #     z = z*1000  # convert meters to mm
    #     new_key = (x, y, z)
    #     if math.isnan(point[0]):
    #         continue
    #     exportDict[new_key] = hsv
    #     ax2.scatter(x, y, z, c=color, marker='o')
    # # Set the axis limits and labels
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')

    # # save the plot as a pdf
    # fig2.savefig('output_image_folder/point_cloud_' + str(count) + '_in_mm.pdf')
    # fig2.savefig('output_image_folder/point_cloud_' + str(count) + '_in_mm.png')

    # colorDict_list.append(colorDict)

    # # Open a file for writing
    # output_dict_folder = 'output_dict_folder/'
    # with open(output_dict_folder + 'exportDict' + str(count) + '.txt', 'w') as f:
    #     for key, value in exportDict.items():
    #         f.write('{}:{}\n'.format(key, value))

    # colorDict.clear()

    # # Show the plots
    # plt.show()
    # # fig2.show()

    # key = cv2.waitKey(delay=1)
    # if key == ord('q'):
    #     break

    listener.release(frames)

    # print(count)
    count += 1

# device.stop()
# device.close()

# sys.exit(0)

# cv2.destroyAllWindows()
