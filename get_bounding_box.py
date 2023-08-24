import cv2
import numpy as np

def get_image_bb( image:np.array=None,
                  image_path:str=None,
                  binary_threshold:int=70,
                  bounding_box_size:tuple=(50,50),) -> (np.array, list):
    '''
    Description:
        獲取單張圖片的 bounding box 資訊
    
    Parameters:

        
    Returns:
        image (np.ndarray): (圖片的寬, 圖片的高, 圖片的通道數)
        bb_info (list): bounding box 的資訊，每個 list 分別是 [x, y, w, h]，x, y 是 bounding box 的左上角
        
    Examples:

    Exception Handling:
        Raises a ValueError if the provided str is empty.
    '''
    if (image_path != None):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 如果照片是彩色的話，就轉換成灰階
    if (len(image.shape) == 3):
        # convert BGR to GRAY
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()  # 如果已經是灰階，直接複製
    
    ret, gray = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)  

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, bounding_box_size)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    contours = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    bb_info = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255))
        coord = [x, y, w, h] 
        bb_info.append(coord)

    return image, bb_info

def get_filter_bb(bb_info:list) -> list:

    # 計算每個 bounding box 的面積
    areas = [box[2]*box[3] for box in bb_info]

    # 找出最大的 bounding box 的面積
    max_area = max(areas)

    # 刪除面積小於最大 bounding box 面積三分之一的 bounding boxes
    filtered_boxes = [box for box, area in zip(bb_info, areas) if area >= max_area/3]

    return filtered_boxes

def crop_images(image, bb_boxes:list)->np.array:
    # 只能放一張圖片進來

    # 判斷bb_boxes是否是一維陣列（代表只有一組bounding box資訊）
    # 你可以透過檢查bb_boxes的第一個元素是否是整數來判斷
    if isinstance(bb_boxes[0], int):
        bb_boxes = [bb_boxes]  # 將一維陣列轉為二維陣列

    # 使用列表推導來裁剪圖片並將結果存放到新的陣列中
    cropped_images = [image[y:y+h, x:x+w] for x, y, w, h in bb_boxes]
    return np.array(cropped_images)


# 將 bounding box 之外的區域都變成白色
def mask_outside_bounding_box(img:np.array, bbox:list, image_path:str = None) -> np.array:
    """
    Masks the area outside the bounding box with white color.

    Parameters:
    - img(np.array): image
    - image_path (str): Path to the input image.
    - bbox (tuple): Bounding box coordinates in the format (x, y, w, h).

    Returns:
    - None
    """

    if image_path != None:
        # 載入照片
        img = cv2.imread(image_path)

    # bounding box資訊
    x, y, w, h = bbox

    # 建立一個與原始照片大小相同的全白圖像
    mask = np.ones_like(img) * 255

    # 將bounding box區域設置為原圖的該部分
    mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    return mask