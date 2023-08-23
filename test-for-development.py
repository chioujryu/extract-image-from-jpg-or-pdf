# convert pdf to jpg
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import numpy as np
from pdf2image import convert_from_path # 如果要使用 convert_from_path 的話，需要下載 poppler 到同個資料夾
from extract_image import show_image,convert_pdf_to_jpg_linux_os
import cv2
import keras_ocr
import math
from inpaint_text import *


images = convert_pdf_to_jpg_linux_os('/opt/cosmo_home/side_project/extract-image-from-jpg-or-pdf/data/pdf/1_pic.pdf',
                                     '/opt/cosmo_home/side_project/extract-image-from-jpg-or-pdf/data/jpg')



# convert BGR to GRAY
gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)    # cv2.COLOR_BGR2GRAY
ret, output1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)     # 如果大於 127 就等於 255，反之等於 0。

print(output1.shape)


# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

img = keras_ocr.tools.read("/opt/cosmo_home/side_project/extract-image-from-jpg-or-pdf/data/jpg/page0.jpg") 
print(img.shape)

#img_text_removed = inpaint_text('traffic-signs.jpg', pipeline)
img_text_removed = inpaint_text(img = img, 
                                pipeline = pipeline)

