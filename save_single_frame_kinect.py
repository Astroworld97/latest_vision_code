# # coding: utf-8

# import numpy as np
# import cv2
# import sys
# from pylibfreenect2 import Freenect2, SyncMultiFrameListener
# from pylibfreenect2 import FrameType, Registration, Frame
# from pylibfreenect2 import createConsoleLogger, setGlobalLogger
# from pylibfreenect2 import LoggerLevel
# from boundingBoxAngleKinect import *

# ### start Kinect initialization###
# try:
#     from pylibfreenect2 import OpenGLPacketPipeline
#     pipeline = OpenGLPacketPipeline()
# except:
#     try:
#         from pylibfreenect2 import OpenCLPacketPipeline
#         pipeline = OpenCLPacketPipeline()
#     except:
#         from pylibfreenect2 import CpuPacketPipeline
#         pipeline = CpuPacketPipeline()
# print("Packet pipeline:", type(pipeline).__name__)

# # Create and set logger
# logger = createConsoleLogger(LoggerLevel.Debug)
# setGlobalLogger(logger)

# fn = Freenect2()  # kinect object
# num_devices = fn.enumerateDevices()
# if num_devices == 0:
#     print("No device connected!")
#     sys.exit(1)

# serial = fn.getDeviceSerialNumber(0)
# device = fn.openDevice(serial, pipeline=pipeline)

# listener = SyncMultiFrameListener(
#     FrameType.Color | FrameType.Ir | FrameType.Depth)

# # Register listeners
# device.setColorFrameListener(listener)
# device.setIrAndDepthFrameListener(listener)

# device.start()
# ### end Kinect initialization###

# # Set up video writer --> might be needed to save the video
# frame = cv2.imread('registered_frame.jpg', cv2.IMREAD_COLOR)
# height, width, channels = frame.shape
# # fps = 30.0  # Frames per second
# # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames
# # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
# # out = cv2.VideoWriter('outputBlue.mp4', fourcc, fps, (width, height))

# # Define the range of the color to be segmented
# ### wood###
# lower1 = np.array([10, 100, 170])
# upper1 = np.array([40, 130, 230])
# ### red paper###
# lower2 = np.array([160, 160, 100])
# upper2 = np.array([190, 200, 240])

# # angle_arr, which will contain all the angles over the span of the camera being open
# angle_arr = []

# # NOTE: must be called after device.start()
# registration = Registration(device.getIrCameraParams(),
#                             device.getColorCameraParams())

# undistorted = Frame(512, 424, 4)
# registered = Frame(512, 424, 4)

# color_data = None
# depth_data = None

# # Open the video stream
# i = 0
# while i < 10000:
#     frames = listener.waitForNewFrame()

#     color = frames["color"]
#     depth = frames["depth"]
#     color_data = color.asarray()
#     color_data = color_data[:, :, :3]
#     depth_data = depth.asarray(np.uint8)
#     # extract the last channel which contains the depth information
#     depth_channel = depth_data[:, :, -1]
#     registration.apply(color, depth, undistorted, registered)
#     registered_data = registered.asarray(np.uint8)
#     registered_data = registered_data[:, :, :3]
#     print(registered_data.shape)
#     print(depth_data.shape)
#     print(depth_channel.shape)
#     registered_frame = registered_data
#     cv2.imwrite('registered_frame.jpg', registered_frame)

#     # Display the original frame and the result
#     cv2.imshow("Original", registered_frame)

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     i += 1
