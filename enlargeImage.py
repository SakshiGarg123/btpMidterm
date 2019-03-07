from PIL import Image, ImageOps
import numpy as np

def add_border(input_image, output_image, border, color=0):

    img = input_image

    if isinstance(border, int) or isinstance(border, tuple):

        bimg = ImageOps.expand(img, border=border, fill=color)

        print (np.array(bimg).shape)

    else:

        raise RuntimeError('Border is not an integer or tuple!')

    bimg.save(output_image)
fil1 = "C:\\Users\\sakshigarg\\Desktop\\Prerna_BTP_Liver disease_2018\\Dataset\\dataset\\"
if __name__ == '__main__':

    for i in range(0, 131):
        img = Image.open(fil1 + 'segliver' + str(i) + 'a.png')
        add_border(img,output_image=fil1 + 'enlarged_liver' + str(i) + 'a.png',border=156)
        img = Image.open(fil1 + 'segliver' + str(i) + 'c.png')
        add_border(img, output_image=fil1 + 'enlarged_liver' + str(i) + 'c.png', border=156)
        img = Image.open(fil1 + 'segliver' + str(i) + 's.png')
        add_border(img, output_image=fil1 + 'enlarged_liver' + str(i) + 's.png', border=156)

        img = Image.open(fil1 + 'seglesion' + str(i) + 'a.png')
        add_border(img, output_image=fil1 + 'enlarged_lesion' + str(i) + 'a.png', border=156)
        img = Image.open(fil1 + 'seglesion' + str(i) + 'c.png')
        add_border(img, output_image=fil1 + 'enlarged_lesion' + str(i) + 'c.png', border=156)
        img = Image.open(fil1 + 'seglesion' + str(i) + 's.png')
        add_border(img, output_image=fil1 + 'enlarged_lesion' + str(i) + 's.png', border=156)

        img = Image.open(fil1 + 'vol' + str(i) + 'a.png')
        add_border(img, output_image=fil1 + 'enlarged_vol' + str(i) + 'a.png', border=156)
        img = Image.open(fil1 + 'vol' + str(i) + 'c.png')
        add_border(img, output_image=fil1 + 'enlarged_vol' + str(i) + 'c.png', border=156)
        img = Image.open(fil1 + 'vol' + str(i) + 's.png')
        add_border(img, output_image=fil1 + 'enlarged_vol' + str(i) + 's.png', border=156)
