source = "C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\Original_Dataset\\"
destination="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\Removedsuperimposition_Dataset\\"
fil1 = "C:\\Users\\sakshigarg\\Desktop\\Prerna_BTP_Liver disease_2018\\Dataset\\dataset\\"
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
for i in range(0,131):
    img = Image.open(source + 'seg'+str(i)+'a.png')
    arr=np.array(img)
    liver = np.copy(arr)
    lesion = np.copy(arr)

    liver[liver<100] = 0
    liver[liver>=100] = 255
    imgsave = Image.fromarray(liver)
    imgsave.save(destination+'segliver'+str(i)+'a.png')

    lesion[lesion<=200] = 0
    lesion[lesion>200] = 255
    imgsave = Image.fromarray(lesion)
    imgsave.save(destination+'seglesion'+str(i)+'a.png')


    img = Image.open(source + 'seg' + str(i) + 'c.png')
    arr = np.array(img)
    liver = np.copy(arr)
    lesion = np.copy(arr)

    liver[liver < 100] = 0
    liver[liver >= 100] = 255
    imgsave = Image.fromarray(liver)
    imgsave.save(destination+ 'segliver' + str(i) + 'c.png')

    lesion[lesion <= 200] = 0
    lesion[lesion > 200] = 255
    imgsave = Image.fromarray(lesion)
    imgsave.save(destination + 'seglesion' + str(i) + 'c.png')



    img = Image.open(source+ 'seg' + str(i) + 's.png')
    arr = np.array(img)
    liver = np.copy(arr)
    lesion = np.copy(arr)

    liver[liver < 100] = 0
    liver[liver >= 100] = 255
    imgsave = Image.fromarray(liver)
    imgsave.save(destination + 'segliver' + str(i) + 's.png')

    lesion[lesion <= 200] = 0
    lesion[lesion > 200] = 255
    imgsave = Image.fromarray(lesion)
    imgsave.save(destination+ 'seglesion' + str(i) + 's.png')