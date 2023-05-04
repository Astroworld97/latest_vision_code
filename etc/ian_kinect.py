# coding: utf-8

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

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

fn = Freenect2() #kinect object
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

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

color_data = None
depth_data = None

# comment this out if you don't want to print the whole array
# np.set_printoptions(threshold=np.inf)

# define the lower and upper bounds of the RGB color range to mask
lower = np.array([88,  126,  158])
upper = np.array([108, 146, 179])

while True:
    frames = listener.waitForNewFrame()

    color = frames["color"]
    depth = frames["depth"]
    color_data = color.asarray()
    depth_data = depth.asarray()
    registration.apply(color, depth, undistorted, registered)
    registered_data = registered.asarray(np.uint8)

    # convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(color_data, cv2.COLOR_BGR2HSV)

    # create a mask for pixels within the specified RGB color range
    color_mask = cv2.inRange(color_data, lower, upper)

    # Filter depth data using a threshold value
    threshold = 1000  # Example threshold value
    depth_mask = np.where(depth_data < threshold, 1, 0)

    # Apply depth mask to color data to extract object
    # object_mask = depth_mask * depth_data

    # print("Color shape:", color_data.shape, "dtype:", color_data.dtype)
    # print("Depth shape:", depth_data.shape, "dtype:", depth_data.dtype)
    # print(color_data)
    # print(depth_data)

    # registration.apply(color, depth, undistorted, registered)

    # cv2.imshow("mask", object_mask)
    # print(object_mask)

    # print(color)

    # cv2.imshow("depth", depth.asarray() / 4500.)
    # cv2.imshow("color", cv2.resize(color.asarray(),
    #                                (int(1920 / 3), int(1080 / 3))))
    # cv2.imshow("registered", registered.asarray(np.uint8))

    # Apply depth mask to registered data to extract object
    depth_extracted = cv2.bitwise_and(registered_data, registered_data, mask=depth_mask)

    # Apply color mask to extracted data
    color_extracted = cv2.bitwise_and(hsv_data, hsv_data, mask=color_mask)

    # Combine color and depth extracted data
    extracted_data = cv2.addWeighted(color_extracted, 0.5, depth_extracted, 0.5, 0)

    cv2.imshow("registered", extracted_data)


    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break


device.stop()
device.close()

sys.exit(0)
