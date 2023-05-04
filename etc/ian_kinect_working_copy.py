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

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

color_data = None
depth_data = None

# define the lower and upper bounds of the RGB color range to mask
mean = np.array([98,  136,  169])
lower = mean-10
upper = mean+10
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

    # convert the frame to the HSV color space
    hsv = cv2.cvtColor(registered_data, cv2.COLOR_BGR2HSV)

    # create a mask for pixels within the specified RGB color range
    color_mask = cv2.inRange(registered_data, lower, upper)

    # Filter depth data using a threshold value
    threshold = 1000  # Example threshold value
    depth_mask = np.where(depth_data < threshold, 1, 0)
    depth_mask = depth_mask.astype(np.uint8)

    # print(registered_data.shape)
    # print(color_mask.shape)
    # print(registered_data.dtype)
    # print(color_mask.dtype)
    # print(depth_data.shape)
    # print(depth_mask.shape)
    # print(depth_data.dtype)
    # print(depth_mask.dtype)

    d1, d2, d3, d4 = cv2.split(depth_data)
    m1, m2, m3, m4 = cv2.split(depth_mask)

    masked_depth_1 = cv2.bitwise_and(d1, d1, mask=m1)
    masked_depth_2 = cv2.bitwise_and(d2, d2, mask=m2)
    masked_depth_3 = cv2.bitwise_and(d3, d3, mask=m3)
    masked_depth_4 = cv2.bitwise_and(d4, d4, mask=m4)

    # # Apply color mask to registered data to extract object
    color_extracted = cv2.bitwise_and(
        registered_data, registered_data, mask=color_mask)

    # Apply depth mask to extracted data
    # depth_extracted = cv2.bitwise_and(depth_data, depth_data, mask=depth_mask)

    depth_extracted = cv2.merge(
        (masked_depth_1, masked_depth_2, masked_depth_3, masked_depth_4))

    # # Combine color and depth extracted data
    depth_extracted = depth_extracted[:, :, :3]  # remove alpha channel
    # print(color_extracted.shape, color_extracted.ndim)
    # print(depth_extracted.shape, depth_extracted.ndim)
    extracted_data = cv2.addWeighted(
        color_extracted, 0.5, depth_extracted, 0.5, 0)

    cv2.imshow("registered", registered_data)
    cv2.imshow("depth original", depth_data)
    cv2.imshow("color", color_extracted)
    cv2.imshow("depth", depth_extracted)
    cv2.imshow("combo", extracted_data)

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break
    # i+=1


device.stop()
device.close()

sys.exit(0)
