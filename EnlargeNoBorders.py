
source = "C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\Removedsuperimposition_Dataset\\"
source2="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\Original_Dataset\\"
destination1="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\EnlargedWithoutBorder_Dataset\\enlarged_liver"
destination2="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\EnlargedWithoutBorder_Dataset\\enlarged_lesion"
destination3="C:\\Users\\sakshigarg\\PycharmProjects\\btpMidterm\\EnlargedWithoutBorder_Dataset\\enlarged_vol"

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


for i in range(0, 131):
    img = Image.open(source + 'segliver' + str(i) + 'a.png')
    img = img.resize((512, 512))
    img.save(destination1+ str(i) + 'a.png')

    img = Image.open(source + 'segliver' + str(i) + 'c.png')
    img = img.resize((512, 512))
    img.save(destination1 + str(i) + 'c.png')

    img = Image.open(source + 'segliver' + str(i) + 's.png')
    img = img.resize((512, 512))
    img.save(destination1 + str(i) + 's.png')


    img = Image.open(source + 'seglesion' + str(i) + 'a.png')
    img = img.resize((512, 512))
    img.save(destination2 + str(i) + 'a.png')

    img = Image.open(source + 'seglesion' + str(i) + 'c.png')
    img = img.resize((512, 512))
    img.save(destination2 + str(i) + 'c.png')

    img = Image.open(source + 'seglesion' + str(i) + 's.png')
    img = img.resize((512, 512))
    img.save(destination2 + str(i) + 's.png')



    img = Image.open(source2 + 'vol' + str(i) + 'a.png')
    img = img.resize((512, 512))
    img.save(destination3 + str(i) + 'a.png')

    img = Image.open(source2+ 'vol' + str(i) + 'c.png')
    img = img.resize((512, 512))
    img.save(destination3 + str(i) + 'c.png')

    img = Image.open(source2 + 'vol' + str(i) + 's.png')
    img = img.resize((512, 512))
    img.save(destination3 + str(i) + 's.png')