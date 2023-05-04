import cv2

# Define a mouse callback function to get the HSV values of the clicked pixel


def get_hsv_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print("HSV values at ({}, {}): {}".format(x, y, hsv[y, x]))


# Load the image
img = cv2.imread("frame.jpg")

# Define a window name and set the mouse callback function
window_name = "Click to get HSV values"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, get_hsv_values)

while True:
    # Display the image
    cv2.imshow(window_name, img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all windows
cv2.destroyAllWindows()
