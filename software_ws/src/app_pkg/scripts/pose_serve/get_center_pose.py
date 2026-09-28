import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_combined_green_center(image_path, gradient_threshold=50, min_area=100):
    """
    计算所有绿色像素分布区域的整体中心点
    参数:
        image_path: 图片路径
        gradient_threshold: 梯度阈值(默认50)
        min_area: 最小区域面积(像素数)，小于此值的区域会被过滤(默认100)
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    img = img[55:,:]
    # 2. 计算梯度并获取绿色边缘区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(grad_mag, gradient_threshold, 255, cv2.THRESH_BINARY)

    # 3. 获取所有绿色像素的分布区域（包括中间非绿色部分）
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # 4. 创建包含所有绿色区域的凸包
    combined_points = np.vstack(valid_contours).squeeze()
    hull = cv2.convexHull(combined_points)
    
    # 5. 计算整体区域中心点（质心）
    M = cv2.moments(hull)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # 6. 在原图上标记结果
    result_img = img.copy()
    # 绘制所有绿色边缘
    cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 1)
    # 绘制包含所有绿色区域的凸包
    cv2.drawContours(result_img, [hull], -1, (255, 0, 0), 1)  # 蓝色边界
    # 标记整体中心点
    cv2.circle(result_img, (cx, cy), 5, (0, 255, 255), -1)  # 黄色中心点
    cv2.putText(result_img, f"Center: ({cx}, {cy})", (cx+10, cy+10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 7. 打印中心点坐标
    print(f"整体绿色分布区域中心点坐标: (x={cx}, y={cy})")

    # 8. 可视化
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Combined Green Region Center'), plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "/home/admind/software_ws/src/app_pkg/config/data1/image/image_173.jpg"
    find_combined_green_center(image_path, 
                             gradient_threshold=43,
                             min_area=10)