from pdf2image import convert_from_path # 如果要使用 convert_from_path 的話，需要下載 poppler 到同個資料夾
import numpy as np

def convert_pdf_to_jpg_windows_os(pdf_path:str, poppler_path:str, save_jpg_path:str=None) -> np.ndarray:
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

    if (save_jpg_path != None):
        for i in range(image_number):
            # Save pages as images in the pdf
            images[i].save(save_jpg_path + '/' + 'page'+ str(i) +'.jpg', 'JPEG')

    # 將 PIL 圖像物件轉換成 NumPy 陣列
    image_array = []
    for i in range(image_number):
        image_array.append(np.array(images[i]))  

    return np.array(image_array)

def convert_pdf_to_jpg_linux_os(pdf_path:str, save_jpg_path:str=None) -> np.ndarray:
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

    if (save_jpg_path != None):
        for i in range(image_number):
            # Save pages as images in the pdf
            images[i].save(save_jpg_path + '/' + 'page'+ str(i) +'.jpg', 'JPEG')

    # 將 PIL 圖像物件轉換成 NumPy 陣列
    image_array = []
    for i in range(image_number):
        image_array.append(np.array(images[i]))  

    return np.array(image_array)