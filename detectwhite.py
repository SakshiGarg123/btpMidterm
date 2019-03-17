import numpy as np
import random
import cv2
#output = np.zeros(image.shape,np.uint8)

image = cv2.imread('4.png',0) # Only for grayscale image
flag = False #found white pixel
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if (image[i][j]==255):
            flag = True #found white
            break
    if(flag):
        break

print(flag)

if(flag):
    cv2.imwrite('white4.png', image)
