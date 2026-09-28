import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles):
    """显示多张图片对比"""
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)  # 修正这里：1行，N列，当前子图索引
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 1. 读取图片并预处理
image_path = '/home/admind/software_ws/src/app_pkg/config/data/image/image_116.jpg'  # 替换为你的图片路径
original = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 检查图片是否成功加载
if original is None:
    raise ValueError("无法读取图片，请检查路径是否正确")

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# 二值化处理（如果是彩色图需要先转灰度）
_, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

# 2. 定义结构元素（核）
kernel_size = (5, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

# 3. 进行膨胀和腐蚀操作
dilated = cv2.dilate(binary, kernel, iterations=1)  # 膨胀
eroded = cv2.erode(binary, kernel, iterations=1)    # 腐蚀

# 4. 显示结果对比
images = [binary, dilated, eroded]
titles = ['Original Binary', 'Dilated (膨胀)', 'Eroded (腐蚀)']
show_images(images, titles)

# 5. 进阶操作：开运算和闭运算
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀后膨胀）
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # 闭运算（先膨胀后腐蚀）

# 显示进阶结果
advanced_images = [binary, opening, closing]
advanced_titles = ['Original', 'Opening (开运算)', 'Closing (闭运算)']
show_images(advanced_images, advanced_titles)