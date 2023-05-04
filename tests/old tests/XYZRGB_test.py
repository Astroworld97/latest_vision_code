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

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

color_data = None
depth_data = None

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
while count < 10:
    frames = listener.waitForNewFrame()
    color = frames["color"]
    depth = frames["depth"]
    color_data = color.asarray()
    color_data = color_data[:, :, :3]
    depth_data = depth.asarray(np.uint8)
    # extract the last channel which contains the depth information
    depth_channel = depth_data[:, :, -1]
    registration.apply(color, depth, undistorted, registered)
    # Get the real-world coordinates of a pixel
    r = 256
    c = 212
    pointXYZ = registration.getPointXYZ(undistorted, r, c)
    pointXYZRGB = registration.getPointXYZRGB(undistorted, registered, r, c)

    print(pointXYZ)
    print(pointXYZRGB)

    listener.release(frames)

    count += 1


# print angles
# print("Wood angle: " + str(wood_angle))
# print("Color angle: " + str(color_angle))
# Show frame and save it
# cv2.imshow("Original", frame)
# cv2.imwrite('bounding_box_test.jpg', frame)
# cv2.waitKey(0)
cv2.destroyAllWindows()
