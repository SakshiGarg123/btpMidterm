import numpy as np
import random
import cv2
import logging
from PIL import Image

source="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\EnlargedWithoutBorder_Dataset\\"
destination="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\RotatedEnlargedWithoutBorder_Dataset\\"

for i in range(0,131):

    print (i)
    print("Liver")

    image = Image.open(source+'enlarged_liver'+str(i)+'a.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_liver'+str(i)+'a.png')
            angle+=60

    image = Image.open(source+'enlarged_liver'+str(i)+'c.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_liver'+str(i)+'c.png')
            angle+=60

    image = Image.open(source+'enlarged_liver'+str(i)+'s.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_liver'+str(i)+'s.png')
            angle+=60


    print("Lesion")

    image = Image.open(source+'enlarged_lesion'+str(i)+'a.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_lesion'+str(i)+'a.png')
            angle+=60

    image = Image.open(source+'enlarged_lesion'+str(i)+'c.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_lesion'+str(i)+'c.png')
            angle+=60

    image = Image.open(source+'enlarged_lesion'+str(i)+'s.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_lesion'+str(i)+'s.png')
            angle+=60


    print("Vol")

    image = Image.open(source+'enlarged_vol'+str(i)+'a.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_vol'+str(i)+'a.png')
            angle+=60

    image = Image.open(source+'enlarged_vol'+str(i)+'c.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_vol'+str(i)+'c.png')
            angle+=60

    image = Image.open(source+'enlarged_vol'+str(i)+'s.png')
    angle = 0
    while(angle<360):
            rotated_image  = image.rotate(angle)
            rotated_image.save(destination+'rotated_'+str(angle)+'_enlarged_vol'+str(i)+'s.png')
            angle+=60
