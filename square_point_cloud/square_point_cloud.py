import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from boundingBoxAngleKinect import *
import random
import matplotlib.pyplot as plt
import colorsys

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
# # cv2.imwrite('frame.jpg', frame)
# height, width, channels = dims_frame.shape
fps = 30.0  # Frames per second
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (512, 424))

# Define the range of the color to be segmented
# ### wood###
# lower1 = np.array([10, 100, 170])
# upper1 = np.array([40, 130, 230])
### blue paper###
lower1 = np.array([83, 166, 173])
upper1 = np.array([113, 196, 233])
### red paper###
lower2 = np.array([160, 160, 100])
upper2 = np.array([190, 200, 240])

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
lower1 = np.array([83, 166, 163])
upper1 = np.array([113, 196, 243])
### red paper###
lower2 = np.array([160, 160, 90])
upper2 = np.array([190, 200, 250])

# angle_arr, which will contain all the angles over the span of the camera being open
angle_arr = []

count = 0
while count < 1000:
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
    # print(registered_data.shape)
    # print(depth_data.shape)
    # print(depth_channel.shape)
    frame = registered_data
    # cv2.imwrite('registered_frame.jpg', registered_frame)
    # frame = registered_frame
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks based on the defined range
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # Apply the mask to the original image
    result1 = cv2.bitwise_and(frame, frame, mask=mask1)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)

    # Find contours of the segmented object
    contoursWood, _ = cv2.findContours(
        mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursColor, _ = cv2.findContours(
        mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contoursWood) == 0:

        print("No contours found for blue segment.")

    if len(contoursColor) == 0:

        print("No contours found for red segment.")

    # declare rectangle dictionary (keys: rects, values: boxes)
    rect_dict_wood = defaultdict(int)
    rect_dict_color = defaultdict(int)

    # Find the bounding boxes of the object
    for cnt in contoursWood:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_dict_wood[rect] = box

    for cnt in contoursColor:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_dict_color[rect] = box

    # Find the two rectangles with the largest area (one is wood color, the other is blue paper color)
    largest_wood_rect, largest_wood_box = findLargestCurrRect(rect_dict_wood)
    largest_color_rect, largest_color_box = findLargestCurrRect(
        rect_dict_color)

    # Adjust angle if necessary
    dims = largest_wood_rect[1]  # width and height of rect
    angle = largest_wood_rect[2]
    w = dims[0]
    h = dims[1]
    if w < h:
        angle = angle + 90 if angle < -45 else angle - 90
    wood_angle = angle
    color_angle = largest_color_rect[2]
    # Draw the bounding boxes on the image
    cv2.imwrite('frame.jpg', frame)
    with open('example.txt', 'w') as f:
        dimensions = str(frame.shape)
        f.write(dimensions + '\n')
        t = str(type(frame))
        f.write(t + '\n')
        dt = str(frame.dtype)
        f.write(dt + '\n')
        dimensions = str(largest_wood_box.shape)
        f.write(dimensions + '\n')
        t = str(type(largest_wood_box))
        f.write(t + '\n')
        dt = str(largest_wood_box.dtype)
        f.write(dt + '\n')
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    cv2.drawContours(frame, [largest_wood_box], 0, (255, 0, 0), 2)
    cv2.drawContours(frame, [largest_color_box], 0, (0, 255, 0), 2)

    # Define the colors for each half of the cylinder
    # color1 = (17/360, 125/255, 210/255)  # HSV values for wood
    color1 = (0.66, 1.0, 1.0)  # HSV values for blue
    color2 = (0.0, 1.0, 1.0)  # HSV values for red

    # plot and store wood points
    for i in range(10):
        x0 = np.linspace(largest_wood_box[0][0], largest_wood_box[1][0], 10)[i]
        y0 = np.linspace(largest_wood_box[0][1], largest_wood_box[1][1], 10)[i]
        z = depth_channel[int(y0), int(x0)]
        # z = random.uniform(-0.435, 0.435)
        # z = random.choice([-0.435, 0.435])
        colorDict[(x0, y0, z)] = color1
        cv2.circle(frame, (int(x0), int(y0)), 2, (0, 0, 255), -1)
        x1 = np.linspace(largest_wood_box[1][0], largest_wood_box[2][0], 10)[i]
        y1 = np.linspace(largest_wood_box[1][1], largest_wood_box[2][1], 10)[i]
        z = depth_channel[int(y1), int(x1)]
        # z = random.uniform(-0.435, 0.435)
        # z = random.choice([-0.435, 0.435])
        colorDict[(x1, y1, z)] = color1
        cv2.circle(frame, (int(x1), int(y1)), 2, (0, 0, 255), -1)
        x2 = np.linspace(largest_wood_box[2][0], largest_wood_box[3][0], 10)[i]
        y2 = np.linspace(largest_wood_box[2][1], largest_wood_box[3][1], 10)[i]
        z = depth_channel[int(y2), int(x2)]
        # z = random.choice([-0.435, 0.435])
        # z = random.uniform(-0.435, 0.435)
        colorDict[(x2, y2, z)] = color1
        cv2.circle(frame, (int(x2), int(y2)), 2, (0, 0, 255), -1)
        x3 = np.linspace(largest_wood_box[3][0], largest_wood_box[0][0], 10)[i]
        y3 = np.linspace(largest_wood_box[3][1], largest_wood_box[0][1], 10)[i]
        z = depth_channel[int(y3), int(x3)]
        # z = random.choice([-0.435, 0.435])
        # z = random.uniform(-0.435, 0.435)
        colorDict[(x3, y3, z)] = color1
        cv2.circle(frame, (int(x3), int(y3)), 2, (0, 0, 255), -1)

    # plot and store color points
    for i in range(10):
        x0 = np.linspace(largest_color_box[0]
                         [0], largest_color_box[1][0], 10)[i]
        y0 = np.linspace(largest_color_box[0]
                         [1], largest_color_box[1][1], 10)[i]
        z = depth_channel[int(y0), int(x0)]
        # z = random.choice([-0.435, 0.435])
        # z = random.uniform(-0.435, 0.435)
        colorDict[(x0, y0, z)] = color2
        cv2.circle(frame, (int(x0), int(y0)), 2, (0, 0, 255), -1)
        x1 = np.linspace(largest_color_box[1]
                         [0], largest_color_box[2][0], 10)[i]
        y1 = np.linspace(largest_color_box[1]
                         [1], largest_color_box[2][1], 10)[i]
        z = depth_channel[int(y1), int(x1)]
        # z = random.choice([-0.435, 0.435])
        # z = random.uniform(-0.435, 0.435)
        colorDict[(x1, y1, z)] = color2
        cv2.circle(frame, (int(x1), int(y1)), 2, (0, 0, 255), -1)
        x2 = np.linspace(largest_color_box[2]
                         [0], largest_color_box[3][0], 10)[i]
        y2 = np.linspace(largest_color_box[2]
                         [1], largest_color_box[3][1], 10)[i]
        z = depth_channel[int(y2), int(x2)]
        # z = random.choice([-0.435, 0.435])
        # z = random.uniform(-0.435, 0.435)
        colorDict[(x2, y2, z)] = color2
        cv2.circle(frame, (int(x2), int(y2)), 2, (0, 0, 255), -1)
        x3 = np.linspace(largest_color_box[3]
                         [0], largest_color_box[0][0], 10)[i]
        y3 = np.linspace(largest_color_box[3]
                         [1], largest_color_box[0][1], 10)[i]
        z = depth_channel[int(y3), int(x3)]
        # z = random.choice([-0.435, 0.435])
        # z = random.uniform(-0.435, 0.435)
        colorDict[(x3, y3, z)] = color2
        cv2.circle(frame, (int(x3), int(y3)), 2, (0, 0, 255), -1)
        # print(i)

    # Label vertices of the bounding boxes
    # for point in largest_wood_box:
    #     cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    #     cv2.putText(frame, " " + str(point[0]) + " , " + str(point[1]),
    #                 tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # for point in largest_color_box:
    #     cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    #     cv2.putText(frame, " " + str(point[0]) + " , " + str(point[1]),
    #                 tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Draw a line from the end of the wood bounding box to the start of the color bounding box.
    endpoints = findEndPointsLine(largest_wood_box, largest_color_box)
    cv2.line(frame, endpoints[0], endpoints[1], (255, 0, 0), 2)
    cv2.imwrite('video_test' + str(count) + '.jpg', frame)
    out.write(frame)
    # Exit on 'q' press
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # Note to self: remember images are indexed as rows and columns. Use r and c to iterate, rather than x and y.
    ### y = rows = r; height###
    ### x = columns = c; width###

    # reconstruct original image as plot
    # Create an empty plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    keys = list(colorDict.keys())
    # print(len(keys))
    # Loop through each pixel and set its color using the given HSV values
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
        ax.scatter(col, 424-row, z, c=color, marker='o')
    # Set the axis limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # save the plot as a pdf
    plt.savefig('point_cloud' + str(count) + '.pdf')
    plt.savefig('point_cloud' + str(count) + '.png')

    colorDict_list.append(colorDict)

    colorDict.clear()
    plt.clf()

    # # Show the plot
    # plt.show()
    listener.release(frames)

    print(count)
    count += 1


# print angles
# print("Wood angle: " + str(wood_angle))
# print("Color angle: " + str(color_angle))
# Show frame and save it
# cv2.imshow("Original", frame)
# cv2.imwrite('bounding_box_test.jpg', frame)
# cv2.waitKey(0)
cv2.destroyAllWindows()
