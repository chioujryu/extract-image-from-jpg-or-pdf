# # convert pdf to jpg
# import matplotlib.pyplot as plt
# import matplotlib.image as img
# import os
# import numpy as np
# import time
# from pdf2image import convert_from_path # 如果要使用 convert_from_path 的話，需要下載 poppler 到同個資料夾
# import cv2
# import keras_ocr
# import math
# from image_processing import *
# from extract_image_from_pdf import *
# from image_display_and_save import *
# from convert_pdf_to_jpg import *
# from get_bounding_box import *
import model_inference

# 初始化
prompt=" Question: What is in this picture? tell me more detail. Answer:"
image_url = "https://img.onl/sDBysN"
image_url = "https://media.istockphoto.com/id/484131476/photo/mixed-breed-dog-selfie-photo.jpg?s=612x612&w=0&k=20&c=QDbV3v6J_Ayh0QHgQ9cqobLxBQwEReEOT9t0KdikoBs="


# # 記錄開始時間
# start_time = time.time()

# # 獲取 raw data
# #raw_image = convert_pdf_to_jpg_windows_os("data/pdf/1_pic.pdf", "poppler-23.08.0/Library/bin")
# raw_images = convert_pdf_to_jpg_linux_os("data/pdf/1_pic.pdf")

# # 將文字抹除
# inpainted_texts_imgs = inpaint_texts(raw_images)

# # 獲取 bb 資訊
# image, bb_infos = get_image_bb( inpainted_texts_imgs[0],
#                                 binary_threshold=70,
#                                 bounding_box_size=(50,50))

# # 過濾 bb
# filtered_bb_infos = get_filter_bb(bb_infos)

# masked_image = mask_outside_bounding_box(raw_images[0], filtered_bb_infos[0])



# edges = canny_processing(   image = masked_image, 
#                             binary_threshold = 70, 
#                             canny_threshold1 = 20, 
#                             canny_threshold2 = 30)

# # 獲取 bb 資訊
# image, bb_infos = get_image_bb( edges,
#                                 binary_threshold=70,
#                                 bounding_box_size=(50,50))

# # 過濾 bb
# filtered_bb_infos = get_filter_bb(bb_infos)

# masked_image = mask_outside_bounding_box(raw_images[0], filtered_bb_infos[0])

# # 儲存圖片
# cv2.imwrite('edge_image.jpg', masked_image)

# # 獲取 bb 資訊
# image, bb_infos = get_image_bb( masked_image,
#                                 binary_threshold=70,
#                                 bounding_box_size=(50,50))

# # 過濾 bb
# filtered_bb_infos = get_filter_bb(bb_infos)

# print("bb_infos =",filtered_bb_infos)

# croped_image = crop_images(image, filtered_bb_infos)
# cv2.imwrite('croped_image.jpg', croped_image)


text = model_inference.image_to_text(image_url=image_url, image=None, prompt=prompt)
print(text)


# # 記錄結束時間
# end_time = time.time()
# # 計算並打印所需時間
# elapsed_time = end_time - start_time
# print(f"The function took {elapsed_time:.2f} seconds to complete.")