# import required libraries
import cv2
import glob
images = glob.glob('chessboard_images/left01.jpg')
img = cv2.imread(images[0])
cv2.imshow(img)
cv2.waitKey(0)
