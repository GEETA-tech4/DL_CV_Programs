import numpy as np
import cv2
from google.colab.patches import cv2_imshow
imgOriginal = cv2.imread("./sample.jpeg")
img = cv2.resize(imgOriginal, (500,500))
cv2_imshow(img)

imgc = img.copy()
croppedImg = imgc[50:250, 0:500]
cv2_imshow(croppedImg)

resizedImg = cv2.resize(img, (1000,700))
cv2_imshow(resizedImg)

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(grayImg)

retval,thresh = cv2.threshold(grayImg, 120, 255, cv2.THRESH_BINARY)
cv2_imshow(thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (255,0,0), 2)
cv2_imshow(contour_img)

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(grayImg)
blob_img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2_imshow(blob_img)

cv2.waitKey(0)
