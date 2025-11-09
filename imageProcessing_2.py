import numpy as np
import cv2

imgOriginal = cv2.imread("./images/sample.jpg")
img = cv2.resize(imgOriginal, (500,500))
cv2.imshow("Original Image",img)
imgc = img.copy()
croppedImg = imgc[50:250, 0:500]
cv2.imshow("Cropped Image",croppedImg)
resizedImg = cv2.resize(img, (1000,700))
cv2.imshow("Resized Image : ",resizedImg)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image",grayImg)
retval,thresh = cv2.threshold(grayImg, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Image", thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (255,0,0), 2)
cv2.imshow("Contour Detection", contour_img)
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(grayImg)
blob_img = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Blob Detection", blob_img)
cv2.waitKey(0)