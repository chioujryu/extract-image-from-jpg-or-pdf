# **Extract image from jpg or pdf**
除了導入的套件要安裝之外，下面這些套件也必須要安裝

* pip install tensorflow==2.8.3
* pip install tensorrt==8.5.1.7
* conda install cuda -c nvidia

## **安裝環境**
### 一般環境創建
```
pip install -r requirements.txt
```
### 使用Anaconda創建環境
```
conda env create -f environment.yml
```
这个命令会读取 environment.yml 文件，并创建一个具有相同依赖的新环境。

## **使用說明**
```
python test.py
```
也可以在 test 程式碼裡面修改要餵給 model 的圖片
