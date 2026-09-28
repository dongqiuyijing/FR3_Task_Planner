#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class GreenCenterDetector:
    def __init__(self):
        rospy.init_node('green_center_detector', anonymous=True)
        self.bridge = CvBridge()
        
        # 参数设置
        self.gradient_threshold = rospy.get_param('~gradient_threshold', 43)
        self.min_area = rospy.get_param('~min_area', 10)
        
        # 订阅图像话题
        self.image_sub = rospy.Subscriber("/camera/cropped_image", Image, self.image_callback)
        
        # # 创建显示窗口
        # cv2.namedWindow('Green Center Detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Green Center Detection', 800, 600)
        
        rospy.loginfo("Green Center Detector Initialized")

    def image_callback(self, msg):
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 处理图像
            result_img, center = self.process_image(cv_image)
            
            # 显示结果
            cv2.imshow('Green Center Detection', result_img)
            cv2.waitKey(1)
            
            rospy.loginfo(f"Detected center at: {center}")

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def process_image(self, img):
        img = img[55:,:]
        # 1. 计算梯度并获取绿色边缘区域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary_mask = cv2.threshold(grad_mag, self.gradient_threshold, 255, cv2.THRESH_BINARY)

        # 2. 获取所有有效轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]
        
        if not valid_contours:
            rospy.logwarn("No valid contours found")
            return img, None
        
        # 3. 创建凸包并计算中心
        combined_points = np.vstack(valid_contours).squeeze()
        hull = cv2.convexHull(combined_points)
        M = cv2.moments(hull)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 4. 标记结果
        result_img = img.copy()
        cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 1)  # 绿色边缘
        cv2.drawContours(result_img, [hull], -1, (255, 0, 0), 1)         # 蓝色凸包
        cv2.circle(result_img, (cx, cy), 5, (0, 255, 255), -1)          # 黄色中心点
        cv2.putText(result_img, f"Center: ({cx}, {cy})", (cx+10, cy+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return result_img, (cx, cy)

    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = GreenCenterDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()