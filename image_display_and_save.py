import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import os
import numpy as np

def show_image(image:np.array, format:str):
    """
    Description:
        展示圖片
    
    Parameters:
        image (np.array):  image 的矩陣
        format (str): 如果 image 是一維矩陣，就寫'gray'，如果是三維矩陣，就寫'rgb'
        
    Examples:
        >>> show_image(image, 'gray')
    """
    plt.axis('on')    # 打開座標軸

    if (format == 'gray'):
        plt.imshow(image, cmap='gray')                                    # 在圖表中繪製圖片    
        plt.show()  
    elif (format == 'rgb'):
        plt.imshow(image)                                    # 在圖表中繪製圖片    
        plt.show() 


def save_images_to_output(images: np.array, base_filename: str = 'bounding_box_image'):
    """
    Description:
        儲存多張圖片
    
    Parameters:
        images (np.array):  四維矩陣，分別是 [圖片張數, 長, 寬, 通道數]
        base_filename (str): image 的圖片名稱
        
    Examples:
        >>> save_images_to_output(images, "image")
    """

    output_directory = "output"
    
    # 檢查output資料夾是否存在，如果不存在，則創建它
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # 判斷images是否是一維陣列（代表只有一張照片資訊）
    # 你可以透過檢查images的第一個元素是否是整數來判斷
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    
    for index, image in enumerate(images):
        # 為每張圖片生成獨特的檔名
        filename = f"{base_filename}_{index}.jpg"
        output_path = os.path.join(output_directory, filename)
        
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


