import cv2
import numpy as np
from google.colab.patches import cv2_imshow

image = np.zeros((500, 700, 3), dtype=np.uint8)
line = cv2.line(image, (50,250), (500,450), (0,255,0), 3)
rectange = cv2.rectangle(image, (50,250), (500,450), (255,0,0),4)
circle = cv2.circle(image,(200,300), 200, (0,0,255), 3)
cv2.ellipse(image, (350, 400), (150, 40), 0, 0, 360, (255, 255, 0), 3)
cv2.putText(image, "Image Annotation", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2_imshow(image)
cv2.waitKey(0)
