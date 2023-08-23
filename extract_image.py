# Extract image from paper

#Import required dependencies
import fitz
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
from pdf2image import convert_from_path # 如果要使用 convert_from_path 的話，需要下載 poppler 到同個資料夾
import numpy as np
import cv2

def extract_image_from_pdf_paper(pdf_file_path: str, images_path: str) -> list:
    """
    Description:
        Extract image from pdf which made from word.
    
    Parameters:
        pdf_file_path (str): pdf的路徑. 
        images_path (str): pdf 轉換成 jpg 後的路徑.
        
    Returns:
        list: [是否成功轉換, 有幾張 JPG].
        
        
    Examples:
        >>> extract_image_from_pdf_paper(data/pdf/1_pic.pdf, data/image)
        [True, 5]

    Exception Handling:
        Raises a ValueError if the provided str is empty.
    """

    #Open PDF file
    pdf_file = fitz.open(pdf_file_path)

    #Get the number of pages in PDF file
    page_nums = len(pdf_file)

    #Create empty list to store images information
    images_list = []

    #Extract all images information from each page
    for page_num in range(page_nums):
        page_content = pdf_file[page_num]
        images_list.extend(page_content.get_images())

    #Raise error if PDF has no images
    if len(images_list)==0:
        raise ValueError(f'No images found in {pdf_file_path}')

    #Save all the extracted images
    for i, img in enumerate(images_list, start=1):
        #Extract the image object number
        xref = img[0]
        #Extract image
        base_image = pdf_file.extract_image(xref)
        #Store image bytes
        image_bytes = base_image['image']
        #Store image extension
        image_ext = base_image['ext']
        #Generate image file name
        image_name = str(i) + '.' + image_ext
        #Save image
        with open(os.path.join(images_path, image_name) , 'wb') as image_file:
            image_file.write(image_bytes)
            image_file.close()



def show_image(image, format):
    """
    Description:
        展示圖片
    
    Parameters:
        image (np.array):  image 的矩陣
        format (str): 如果 image 是一維矩陣，就寫'gray'，如果是三維矩陣，就寫'rgb'
        
    Examples:
        >>> show_image(image, 'gray')
    """
    if (format == 'gray'):
        plt.imshow(image, cmap='gray')                                    # 在圖表中繪製圖片    
        plt.show()  
    elif (format == 'rgb'):
        plt.imshow(image)                                    # 在圖表中繪製圖片    
        plt.show() 

def convert_pdf_to_jpg_windows_os(pdf_path:str, poppler_path:str, save_jpg_path:str) -> np.ndarray:
    '''
    Description:
        將 pdf 轉換成圖片，這個function只能給 windows 作業系統用
    
    Parameters:
        pdf_path (str): pdf的路徑. 
        poppler_path (str): 去網路上下載 poppler_path，並且解壓縮，並將此路徑放到引數 'poppler-23.08.0/Library/bin'
        
    Returns:
        np.ndarray: (幾張圖片, 圖片的寬, 圖片的高, 圖片的通道數)
        
    Examples:
        >>> image_array = convert_pdf_to_jpg('data/pdf/Auxiliary Tasks in Multi-task Learning.pdf', 'poppler-23.08.0/Library/bin')
        >>> print(image_array.shape)
        (8, 2200, 1700, 3)

    Exception Handling:
        Raises a ValueError if the provided str is empty.
    '''

    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf_path, poppler_path = poppler_path)   # r'poppler-23.08.0/Library/bin'
    
    # how many image
    image_number = len(images)

    for i in range(image_number):
        # Save pages as images in the pdf
        images[i].save(save_jpg_path + '/' + 'page'+ str(i) +'.jpg', 'JPEG')

    # 將 PIL 圖像物件轉換成 NumPy 陣列
    image_array = []
    for i in range(image_number):
        image_array.append(np.array(images[i]))  

    return image_array

def convert_pdf_to_jpg_linux_os(pdf_path:str, save_jpg_path:str) -> np.ndarray:
    '''
    Description:
        將 pdf 轉換成圖片，這個function只能給 windows 作業系統用
    
    Parameters:
        pdf_path (str): pdf的路徑. 
        
    Returns:
        np.ndarray: (幾張圖片, 圖片的寬, 圖片的高, 圖片的通道數)
        
    Examples:
        >>> image_array = convert_pdf_to_jpg('data/pdf/Auxiliary Tasks in Multi-task Learning.pdf', 'poppler-23.08.0/Library/bin')
        >>> print(image_array.shape)
        (8, 2200, 1700, 3)

    Exception Handling:
        Raises a ValueError if the provided str is empty.
    '''

    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf_path)   # r'poppler-23.08.0/Library/bin'
    
    # how many image
    image_number = len(images)

    for i in range(image_number):
        # Save pages as images in the pdf
        images[i].save(save_jpg_path + '/' + 'page'+ str(i) +'.jpg', 'JPEG')

    # 將 PIL 圖像物件轉換成 NumPy 陣列
    image_array = []
    for i in range(image_number):
        image_array.append(np.array(images[i]))  

    return image_array


def save_one_image(save_path: str, image_name: str, image: np.array) -> bool:
    # 構造完整的儲存路徑 
    save_file = os.path.join(save_path, image_name)
    # 寫入圖片
    image.save(save_file)

    return True



