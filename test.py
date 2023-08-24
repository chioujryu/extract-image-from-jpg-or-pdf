# convert pdf to jpg
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import numpy as np
import time
from pdf2image import convert_from_path # 如果要使用 convert_from_path 的話，需要下載 poppler 到同個資料夾
import cv2
import keras_ocr
import math
from inpaint_text import *
from extract_image_from_pdf import *
from image_display_and_save import *
from convert_pdf_to_jpg import *
from get_bounding_box import *


# 記錄開始時間
start_time = time.time()

raw_image = convert_pdf_to_jpg_windows_os("data/pdf/1_pic.pdf", "poppler-23.08.0/Library/bin")

img = cv2.imread("output/bounding_box_image_0.jpg")

inpaint_texts_img = inpaint_texts(img)

image, bb_infos = get_image_bb(inpaint_texts_img[0],
                  binary_threshold=100,
                  bounding_box_size=(50,50))

filtered_bb_infos = get_filter_bb_boxes(bb_infos)

croped_image = crop_images(raw_image[0], filtered_bb_infos)

show_image(croped_image[0], "rgb")



'''
# convert BGR to GRAY
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, gray = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)     # 如果大於 127 就等於 255，反之等於 0。

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, bounding_box_size)
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

contours = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

bb_info = []

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255))
    coord = [x, y, w, h] 
    bb_info.append(coord)

'''






# 記錄結束時間
end_time = time.time()
# 計算並打印所需時間
elapsed_time = end_time - start_time
print(f"The function took {elapsed_time:.2f} seconds to complete.")