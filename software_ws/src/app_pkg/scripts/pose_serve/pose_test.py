import cv2
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class get_center_pose:
    def __init__(self):
        rospy.init_node('get_center_pose', anonymous=True)
        self.image_sub = rospy.Subscriber("/camera/cropped_image", Image, self.image_callback)
        self.bridge = CvBridge()
    def image_callback(self,msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.find_combined_green_center(cv_image,43,12)
    def find_combined_green_center(self, img, gradient_threshold=50, min_area=100):
        """
        计算所有绿色像素分布区域的整体中心点（使用OpenCV持续显示）
        参数:
            img: 输入图像(numpy数组)
            gradient_threshold: 梯度阈值(默认50)
            min_area: 最小区域面积(像素数)，小于此值的区域会被过滤(默认100)
        """
        # 1. 计算梯度并获取绿色边缘区域
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        grad_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary_mask = cv2.threshold(grad_mag, gradient_threshold, 255, cv2.THRESH_BINARY)

        # 2. 获取所有有效轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # 初始化结果图像
        result_img = img.copy()
        
        if len(valid_contours) > 0:
            # 3. 创建凸包并计算中心
            combined_points = np.vstack(valid_contours).squeeze()
            hull = cv2.convexHull(combined_points)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                620
                # 4. 标记结果
                cv2.drawContours(result_img, valid_contours, -1, (0, 255, 0), 1)  # 绿色边缘
                cv2.drawContours(result_img, [hull], -1, (255, 0, 0), 1)         # 蓝色凸包
                cv2.circle(result_img, (cx, cy), 5, (0, 255, 255), -1)           # 黄色中心点
                cv2.putText(result_img, f"Center: ({cx}, {cy})", (cx+10, cy+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                rospy.loginfo(f"Detected center at: ({cx}, {cy})")
            else:
                rospy.logwarn("No valid center detected (M['m00'] == 0)")
        else:
            rospy.logwarn("No valid contours found")
        
        # 5. 显示结果
        cv2.imshow('Green Center Detection', result_img)
        cv2.waitKey(1)  # 必须调用waitKey才能正常显示
        
        return result_img

def main():
    get_center = get_center_pose()
    rospy.spin()
if __name__ == "__main__":
    main()