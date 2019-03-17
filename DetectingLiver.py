import numpy as np
import random
import cv2
import logging
#output = np.zeros(image.shape,np.uint8)
from PIL import Image

source="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\EnlargedWithoutBorder_Dataset\\"
destination="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\LesionPresent\\"

logging.basicConfig(filename="LesionPresent.log",
                    format='%(asctime)s %(message)s',level=logging.INFO)
for i in range(0,131):
    image = Image.open(source+'enlarged_lesion'+str(i)+'a.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flag = False  # found white pixel

    for l in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (arr[l][j] == 255):
                flag = True  # found white
                break
        if (flag):
            break
    print(flag)

    if (flag):
        # cv2.imwrite('white4.png', image)
        logging.info(source+'enlarged_lesion'+str(i)+'a.png')

    image = Image.open(source + 'enlarged_lesion' + str(i) + 'c.png')  # Only for grayscale image pick up enalrged liver
    arr = np.array(image)
    flag = False  # found white pixel
    for l in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (arr[l][j] == 255):
                flag = True  # found white
                break
        if (flag):
            break
    print(flag)

    if (flag):
        # cv2.imwrite('white4.png', image)
        logging.info(source + 'enlarged_lesion' + str(i) + 'c.png')

    image = Image.open(source + 'enlarged_lesion' + str(i) + 's.png')  # Only for grayscale image pick up enalrged liver
    arr = np.array(image)
    flag = False  # found white pixel
    for l in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (arr[l][j] == 255):
                flag = True  # found white
                break
        if (flag):
            break
    print(flag)

    if (flag):
        # cv2.imwrite('white4.png', image)
        logging.info(source + 'enlarged_lesion' + str(i) + 's.png')
