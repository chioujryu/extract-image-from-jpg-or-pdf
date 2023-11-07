# # pip install accelerate
# import requests
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import torch
# print("Pytorch version is",torch.__version__)
# print("Are we using a GPU?",torch.cuda.is_available())

# torch.cuda.empty_cache()

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
# )

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# question = "how many dogs are in the picture?"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True).strip())


from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import numpy as np
import os
print("Pytorch version is",torch.__version__)
print("Are we using a GPU?",torch.cuda.is_available())


def image_to_text(image_url:str=None, image:np.array=None, prompt:str=None):

    print("function running")
    print("image_url =", image_url)
    print("image =", image)
    print("prompt =", prompt)

    print("Pytorch version is",torch.__version__)
    print("Are we using a GPU?",torch.cuda.is_available())

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =",device)
    #檢查是否有可用的GPU如果有設置為cuda，沒有則為cpu

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    #創建一個model，from_pretrained載入" "預訓練模型的權重參數

    model.to(device)#將模型移動到指定的設備上的程式碼>:(

    if (image==None):
        # image_url = "https://img.onl/sDBysN"   # https://img.onl/sDBysN  https://img.onl/1rQz9r
        image = Image.open(requests.get(image_url, stream=True).raw)

    print("image =", image)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    #將圖像傳遞給processor進行預處理，return_tensors指定回傳PyTorch的張量(序列?)形式
    #將預處理後的張量移動到指定設備，指定資料類型為torch.float16

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    #generate將input解包為關鍵字參數傳遞給model，生成文本序列
    # **input???

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    #batch_decode將多個生成的文本序列，批量解碼為可讀的文本
    #[0]第一個解碼文本
    #strip()去除文本两端的空格

    return generated_text