import numpy as np
import random
import cv2
import logging
from PIL import Image

source="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\EnlargedWithoutBorder_Dataset\\"
destination="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\FlippedEnlargedWithoutBorder_Dataset\\"

for i in range(0,131):

    print (i)
    print("Liver")

    image = Image.open(source+'enlarged_liver'+str(i)+'a.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_liver'+str(i)+'a.png')

    image = Image.open(source+'enlarged_liver'+str(i)+'c.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_liver'+str(i)+'c.png')

    image = Image.open(source+'enlarged_liver'+str(i)+'s.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_liver'+str(i)+'s.png')


    print("Lesion")

    image = Image.open(source+'enlarged_lesion'+str(i)+'a.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_lesion'+str(i)+'a.png')

    image = Image.open(source+'enlarged_lesion'+str(i)+'c.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_lesion'+str(i)+'c.png')

    image = Image.open(source+'enlarged_lesion'+str(i)+'s.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_lesion'+str(i)+'s.png')

    print("Vol")

    image = Image.open(source+'enlarged_vol'+str(i)+'a.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_vol'+str(i)+'a.png')

    image = Image.open(source+'enlarged_vol'+str(i)+'c.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_vol'+str(i)+'c.png')

    image = Image.open(source+'enlarged_vol'+str(i)+'s.png')
    arr = np.array(image)# Only for grayscale image pick up enalrged liver
    flipped_arr = np.fliplr(arr)
    flipped_image  = Image.fromarray(flipped_arr)
    flipped_image.save(destination+'flipped_enlarged_vol'+str(i)+'s.png')
