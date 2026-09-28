#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
class RobustImageSimilarityAnalyzer:
    def __init__(self):
        rospy.init_node('node_image_2_1')
        self.pub = rospy.Publisher('chat', String, queue_size=10)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("chat2", String, self.run_callback, queue_size=1)
        self.start = False
        while not self.start:
            pass
        # 加载目标图像（带安全检查）
        self.target_img = self.load_target_image(
            "/home/admind/software_ws/src/app_pkg/config/data/image/image_930.jpg"
        )
        if self.target_img is None:
            rospy.signal_shutdown("目标图像加载失败")
            return
            
        # 订阅图像话题（带连接状态检查）
        self.image_sub = rospy.Subscriber("/camera/cropped_image", Image, self.image_callback,queue_size=5)

        rospy.loginfo("节点初始化完成，等待图像数据...")
    def run_callback(self,msg):
        if msg.data == "start":
            self.start = True
    def load_target_image(self, path):
        """安全加载目标图像"""
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("无法读取图像文件")
                
            # 调整尺寸并标准化
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
        except Exception as e:
            rospy.logerr(f"加载目标图像错误: {str(e)}")
            return None

    def image_callback(self, msg):
        """处理图像回调"""
        try:
            # 转换图像消息
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            except CvBridgeError as e:
                rospy.logerr(f"图像转换失败: {str(e)}")
                return
                
            # 安全检查
            if cv_img.size == 0:
                rospy.logwarn("收到空图像")
                return
                
            # 预处理当前帧
            current_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX)

            
            # 计算相似度
            similarity = self.calculate_similarity(current_img)
            
            # 显示结果
            self.display_results(current_img, similarity)
            
        except Exception as e:
            rospy.logerr(f"处理错误: {str(e)}")

    def calculate_similarity(self, current_img):
        """计算图像相似度（兼容无skimage的情况）"""
        try:
            # 尝试使用skimage
            from skimage.metrics import structural_similarity as ssim
            score, diff = ssim(
                self.target_img, current_img,
                win_size=11,
                data_range=255,
                full=True
            )
            return score, diff
        except ImportError:
            # 回退到OpenCV实现
            rospy.logwarn("skimage不可用，使用OpenCV替代方案")
            return self.opencv_ssim(self.target_img, current_img), None

    def opencv_ssim(self, img1, img2):
        """OpenCV实现的SSIM近似"""
        # 转换为浮点型
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # 计算均值
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        # 计算方差和协方差
        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1**2
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2**2
        sigma12 = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1*mu2
        
        # SSIM计算
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)

    def display_results(self, current_img, similarity):
        """显示比较结果"""
        score, diff = similarity if isinstance(similarity, tuple) else (similarity, None)
        
        # 计算并打印差异度 (新增部分)
        diff_value = 1.0 - score
        score_limit = rospy.get_param("score_limit")
        if score > score_limit :
            # print("2-1号到达目标点,目标差异值为：",score)
            msg = String()
            msg.data = "go"
            self.pub.publish(msg)
        # print(f"3差异度: {score:.4f}")  # 打印4位小数的差异度
        
        # # 创建显示图像
        # display = np.hstack([
        #     cv2.cvtColor(self.target_img, cv2.COLOR_GRAY2BGR),
        #     cv2.cvtColor(current_img, cv2.COLOR_GRAY2BGR)
        # ])
        
        # # 添加差异图（如果可用）
        # if diff is not None:
        #     diff_vis = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     display = np.hstack([display, diff_vis])
        
        # # 添加文字信息
        # text = f"Similarity: {score:.3f}"
        # cv2.putText(display, text, (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # # 显示图像
        # cv2.imshow("aim_image", display)
        # cv2.waitKey(1)
    def shutdown_hook(self):
        """节点关闭时的清理工作"""
        cv2.destroyAllWindows()
        rospy.loginfo("节点已安全关闭")

if __name__ == '__main__':
    try:
        node = RobustImageSimilarityAnalyzer()
        rospy.on_shutdown(node.shutdown_hook)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass