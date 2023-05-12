# # imgRows = image.shape[0] # height
# # imgCols = image.shape[1] # width
# # channels = image.shape[2] # number of channels

### Surplus Code###
# print(h[1879,2946])
# print(s[0,2946])
# print(v[0,2946])
# print(h.shape) #(1880, 2947)
# print(s.shape) #(1880, 2947)
# print(v.shape) #(1880, 2947)

# Get pixel value
# pixel = image[x, y]


# # Convert point cloud to numpy array
# point_cloud = np.array(point_cloud)
# print(point_cloud)


#  from PIL import Image
# import colorsys

# def rgb_of_pixel(img_path, x, y):
#     img = Image.open(img_path).convert('RGB')
#     r, g, b = img.getpixel((x,y))
#     a = (r, g, b)
#     return a

# img = "flag.png"
# pixel_rgb = rgb_of_pixel(img, 5, 5)
# init_hsv = colorsys.rgb_to_hsv(pixel_rgb[0]/255., pixel_rgb[1]/255., pixel_rgb[0]/255.)
# actual_hsv = (init_hsv[0]*360, init_hsv[1]*100, init_hsv[2]*100)
# print(init_hsv)
# print(actual_hsv)

# converts from opencv's hsv to regular hsv you can input into a regular HSV to RGB converter
# print(convert_from_opencv_hsv_to_regular_hsv(hsv1))
# print(convert_from_opencv_hsv_to_regular_hsv(hsv2))

# def convert_from_opencv_hsv_to_regular_hsv(opencv_hsv):
#     regular_hsv = (opencv_hsv[0]*360/180), (opencv_hsv[1]*100/255), (opencv_hsv[2]*100/255)
#     return regular_hsv

# comment this out if you don't want to print the whole array
# np.set_printoptions(threshold=np.inf)

# Initialize the HSV image
# hsv = np.zeros((424, 512, 3), np.uint8)

# Define the mouse click callback function
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print("HSV value at ({},{}): {}".format(x, y, hsv[y, x]))

# # Set the mouse click callback function for the Kinect window
# cv2.namedWindow("registered")
# cv2.setMouseCallback("registered", mouse_callback)

# import cv2

# import numpy as np

# from collections import defaultdict
# import sys
# from pylibfreenect2 import Freenect2, SyncMultiFrameListener
# from pylibfreenect2 import FrameType, Registration, Frame
# from pylibfreenect2 import createConsoleLogger, setGlobalLogger
# from pylibfreenect2 import LoggerLevel


# ### Start initialization of Kinect camera ###

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

# # NOTE: must be called after device.start()
# registration = Registration(device.getIrCameraParams(),
#                             device.getColorCameraParams())

# undistorted = Frame(512, 424, 4)
# registered = Frame(512, 424, 4)

# color_data = None
# depth_data = None

# ### End initialization of Kinect camera ###


# # Define the range of the color to be segmented (note: the color values are in HSV, not RGB)

# lower = np.array([10, 100, 170])

# upper = np.array([40, 130, 230])


# # angle_arr, which will contain all the angles over the span of the camera being open

# angle_arr = []


# # count for saving a frame

# count = 0


# while True:

#     # Capture frame-by-frame

#     frames = listener.waitForNewFrame()

#     color = frames["color"]
#     depth = frames["depth"]
#     color_data = color.asarray()
#     color_data = color_data[:, :, :3]
#     depth_data = depth.asarray(np.uint8)
#     registration.apply(color, depth, undistorted, registered)
#     registered_data = registered.asarray(np.uint8)
#     registered_data = registered_data[:, :, :3]

#     # Convert the frame to the HSV color space

#     hsv = cv2.cvtColor(registered_data, cv2.COLOR_BGR2HSV)

#     # Create a mask based on the defined range

#     mask = cv2.inRange(hsv, lower, upper)

#     # Apply the mask to the original image

#     result = cv2.bitwise_and(registered_data, registered_data, mask=mask)

#     # Find contours of the segmented object

#     contours, _ = cv2.findContours(
#         mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # If no contours are found, do nothing. I added this because I was getting errors when there were no contours, which was usually at the beginning of the video, aka the beginning of the while loop
#     if len(contours) == 0:
#         continue

#     # declare rectangle dictionary (keys: rects, values: boxes)

#     rect_dict = defaultdict(int)

#     # Find the bounding box of the object

#     for cnt in contours:

#         rect = cv2.minAreaRect(cnt)

#         box = cv2.boxPoints(rect)

#         box = np.int0(box)

#         rect_dict[rect] = box

#     # Find the rectangle with the largest area

#     max_area = -1

#     # initializes largest_rect to be the first rect in the keys array of rect_dict
#     largest_rect = next(iter(rect_dict))

#     for rect in rect_dict.keys():

#         dims = rect[1]  # width and height of curr rect

#         w = dims[0]

#         h = dims[1]

#         area = w * h

#         if area > max_area:

#             max_area = area

#             largest_rect = rect

#     rect = largest_rect

#     box = rect_dict[rect]

#     # Adjust angle if necessary

#     dims = rect[1]  # width and height of rect

#     angle = rect[2]

#     w = dims[0]

#     h = dims[1]

#     if w < h:

#         angle = angle + 90 if angle < -45 else angle - 90

#     # Draw the bounding box on the image

#     reg = cv2.UMat(registered_data)

#     cv2.drawContours(reg, [box], 0, (0, 0, 255), 2)

#     # Calculate the angle of the bounding box

#     angle = rect[2]

#     # Print the angle on the image and store angle in angle_arr

#     cv2.putText(reg, "Angle: {:.2f}".format(
#         angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # append current angle to angle_arr

#     angle_arr.append(angle)

#     # Display the original frame and the result

#     img = reg.get()
#     cv2.imshow("Original", registered_data)

#     cv2.imshow("Result", result) #Uncomment this line to see what is being segmented by the code

#     # save the frame numbered at count

#     if count == 300:

#         cv2.imwrite("example_frame.jpg", registered_data)

#     count += 1

#     # Exit if 'q' is pressed

#     if cv2.waitKey(1) & 0xFF == ord('q'):

#         break


# # Release the capture and destroy all windows

# device.stop()
# device.close()
# cv2.destroyAllWindows()
# sys.exit(0)

# # Uncomment line below to print angle_arr, which contains all the angles over the span of the camera being open

# # print(angle_arr)

### start global cartesian axes###
# # Define the origin of the plane
# origin = (frame.shape[1]//2, frame.shape[0]//2) #frame height = frame.shape[0]; frame width = frame.shape[1]

# # Draw horizontal line
# cv2.line(frame, (0, int(origin[1])), (frame.shape[1], int(origin[1])), (0, 0, 0), 1)

# # Draw vertical line
# cv2.line(frame, (int(origin[0]), 0), (int(origin[0]), frame.shape[0]), (0, 0, 0), 1)
### end global cartesian axes###

# Read the first frame
# first_frame = depth_data

# # Check if the frame was successfully read
# if first_frame is not None:
#     # Save the frame as an image file
#     cv2.imwrite('first_frame.jpg', first_frame)
#     print('First frame saved successfully.')
# else:
#     print('Failed to read the first frame.')

# Display the original frame and the result
# cv2.imshow("Original", frame)

# save the first frame
# cv2.imwrite('frame.jpg', frame)
# print('Frame saved successfully.')

# Write the modified frame to the video file
# out.write(frame)


# Open the video stream
# i = 0
# while i < 1:
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
#     # cv2.imshow("Original", registered_frame)

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     i += 1

# ### wood###
# lower1 = np.array([10, 100, 170])
# upper1 = np.array([40, 130, 230])

# with open('example.txt', 'w') as f:
#         dimensions = str(frame.shape)
#         f.write(dimensions + '\n')
#         t = str(type(frame))
#         f.write(t + '\n')
#         dt = str(frame.dtype)
#         f.write(dt + '\n')
#         dimensions = str(largest_wood_box.shape)
#         f.write(dimensions + '\n')
#         t = str(type(largest_wood_box))
#         f.write(t + '\n')
#         dt = str(largest_wood_box.dtype)
#         f.write(dt + '\n')

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


# keys = list(colorDict.keys())
# print(len(keys))


# print angles
# print("Wood angle: " + str(wood_angle))
# print("Color angle: " + str(color_angle))
# Show frame and save it
# cv2.imshow("Original", frame)
# cv2.imwrite('bounding_box_test.jpg', frame)

# ### wood###
# lower1 = np.array([10, 100, 170])
# upper1 = np.array([40, 130, 230])

# Define the range of the color to be segmented
# ### blue paper###
# lower1 = np.array([83, 166, 173])
# upper1 = np.array([113, 196, 233])
# ### red paper###
# lower2 = np.array([160, 160, 100])
# upper2 = np.array([190, 200, 240])

# color1 = (17/360, 125/255, 210/255)  # HSV values for wood

# Adjust angle if necessary
# dims = largest_blue_rect[1]  # width and height of rect
# angle = largest_blue_rect[2]
# w = dims[0]
# h = dims[1]
# if w < h:
#     angle = angle + 90 if angle < -45 else angle - 90
# blue_angle = angle
# red_angle = largest_red_rect[2]
# # Draw the bounding boxes on the image
# cv2.imwrite('frame.jpg', frame)

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames

# print(registered_data.shape)
# print(depth_data.shape)
# print(depth_channel.shape)

# def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
#     z = np.linspace(0, height_z, 50)
#     theta = np.linspace(0, 2*np.pi, 50)
#     theta_grid, z_grid = np.meshgrid(theta, z)
#     x_grid = radius*np.cos(theta_grid) + center_x
#     y_grid = radius*np.sin(theta_grid) + center_y
#     return x_grid, y_grid, z_grid