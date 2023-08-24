import cv2
import keras_ocr
import math
import numpy as np
import matplotlib.pyplot as plt
import os



#General Approach.....
#Use keras OCR to detect text, define a mask around the text, and inpaint the
#masked regions to remove the text.
#To apply the mask we need to provide the coordinates of the starting and 
#the ending points of the line, and the thickness of the line

#The start point will be the mid-point between the top-left corner and 
#the bottom-left corner of the box. 
#the end point will be the mid-point between the top-right corner and the bottom-right corner.
#The following function does exactly that.
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

'''
#Main function that detects text and inpaints. 
#Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img:np.array=None, 
                 img_path:str = "") -> np.array:
    # read the image 
    if (img_path != ""):
        img = keras_ocr.tools.read(img_path) 

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()
    
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([img])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        #For the line thickness, we will calculate the length of the line between 
        #the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)   # INPAINT_TELEA    # INPAINT_NS
                 
    return np.array(inpainted_img)
'''



# Main function that detects text and inpaints. 
# Inputs are the images array and directory path containing images
def inpaint_texts(imgs:list=None, 
                  img_dir:str=None) -> np.array:
    '''
    Description:
        將圖片的文字抹除
    
    Parameters:
        imgs (list): 像是[8, 640, 480, 3]，意思是有8張640x480的彩色圖片
        img_dir (str): 存放很多 image 圖片的資料夾
        
    Returns:
        np.ndarray: [幾張圖片, 圖片的寬, 圖片的高, 圖片的通道數]
        
    Examples:
        >>> images = inpaint_texts(imgs = imags)
        >>> print(images.shape)
        [8, 2200, 1700, 3]

    Exception Handling:
        Raises a ValueError if the provided str is empty.
    '''
    inpaint_texts_imgs = []

    # Load images from directory if provided
    if img_dir:
        image_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        for image_file in image_files:
            image_path = os.path.join(img_dir, image_file)
            imgs.append(keras_ocr.tools.read(image_path))

    # Check if imgs is three dimensional and expand its dimensions if it is
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0)

    for img in imgs:
        # keras-ocr will automatically download pretrained
        # weights for the detector and recognizer.
        pipeline = keras_ocr.pipeline.Pipeline()
        
        # Recogize text (and corresponding regions)
        # Each list of predictions in prediction_groups is a list of
        # (word, box) tuples. 
        prediction_groups = pipeline.recognize([img])
        
        #Define the mask for inpainting
        mask = np.zeros(img.shape[:2], dtype="uint8")
        for box in prediction_groups[0]:
            x0, y0 = box[1][0]
            x1, y1 = box[1][1] 
            x2, y2 = box[1][2]
            x3, y3 = box[1][3] 
            
            x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
            x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
            
            #For the line thickness, we will calculate the length of the line between 
            #the top-left corner and the bottom-left corner.
            thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
            
            #Define the line and inpaint
            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
            thickness)

        inpainted_img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)   # INPAINT_TELEA    # INPAINT_NS
        inpaint_texts_imgs.append(inpainted_img)

                 
    return np.array(inpaint_texts_imgs)


#cv2.imwrite('text_removed_image.jpg', cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))
