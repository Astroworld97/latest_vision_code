# coding: utf-8

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
from boundingBoxAngleKinect import *

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
frame = cv2.imread('registered_frame.jpg', cv2.IMREAD_COLOR)
# cv2.imwrite('frame.jpg', frame)
height, width, channels = frame.shape
fps = 30.0  # Frames per second
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
#     *'MJPG'), fps, (width, height))

# Define the range of the color to be segmented
### wood###
lower1 = np.array([10, 100, 170])
upper1 = np.array([40, 130, 230])
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

# Open the video stream
i = 0
while True:
    frames = listener.waitForNewFrame()

    color = frames["color"]
    depth = frames["depth"]
    color_data = color.asarray()
    color_data = color_data[:, :, :3]
    depth_data = depth.asarray(np.uint8)
    registration.apply(color, depth, undistorted, registered)
    registered_data = registered.asarray(np.uint8)
    registered_data = registered_data[:, :, :3]

    frame = registered_data
    # frame = cv2.imread('frame.jpg')
    # cv2.imwrite('frame.jpg', frame)

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

    if len(contoursWood) == 0 or len(contoursColor) == 0:

        print("No contours found for either or both wood and color segments.")

        # Display the original frame and the result
        # cv2.imshow("Original", frame)

        # save the first frame
        # cv2.imwrite('frame.jpg', frame)
        # print('Frame saved successfully.')

        # Write the modified frame to the video file
        # out.write(frame)

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

    # Draw the bounding boxes on the image
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    cv2.drawContours(frame, [largest_wood_box], 0, (0, 0, 255), 2)
    cv2.drawContours(frame, [largest_color_box], 0, (0, 255, 0), 2)

    # Draw a line from the end of the wood bounding box to the start of the color bounding box.
    endpoints = findEndPointsLine(largest_wood_box, largest_color_box)
    cv2.line(frame, endpoints[0], endpoints[1], (255, 0, 0), 2)

    # # Calculate the angle of the bounding box
    # angle = largest_wood_rect[2]

    # # Print the angle on the image and store angle in angle_arr
    # cv2.putText(frame, "Angle: {:.2f}".format(
    #     angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # append current angle to angle_arr
    # angle_arr.append(angle)

    # Display the original frame and the result
    # cv2.imshow("Original", frame)
    # cv2.imshow("Result1", result1) #Uncomment this line to see what is being segmented by the code
    # cv2.imshow("Result2", result2) #Uncomment this line to see what is being segmented by the code

    # Write the modified frame to the video file
    # out.write(frame)

    # Display the frame
    # saved in the file
    # cv2.imshow('Frame', frame)

    # uncomment to save the frame numbered at count
    # if count == 300:
    #     cv2.imwrite("example_frame.jpg", frame)
    # count+=1
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1


# Uncomment line below to print angle_arr, which contains all the angles over the span of the camera being open
# print(angle_arr)

# Release video writer and exit
# out.release()
cv2.destroyAllWindows()
