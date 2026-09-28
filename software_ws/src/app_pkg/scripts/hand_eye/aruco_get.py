import numpy as np
import time
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import time
# LHR2023-10-19

# 相机内参
import yaml
# print(dir(aruco))
file_path = ("/home/admind/control_ws/src/realsense_control_pkg/config/标定文件.yaml")
# file_path = ("/home/pi/PycharmProjects/pythonProject/opencv_test/变焦相机标定参数.yaml")
###加载文件路径###

with open(file_path, "r") as file:
    parameter = yaml.load(file.read(), Loader=yaml.Loader)
    mtx = parameter['camera_matrix']
    dist = parameter['dist_coeff']
    camera_u = parameter['camera_u']
    camera_v = parameter['camera_v']
    mtx = np.array(mtx)
    dist = np.array(dist)

# 打开笔记本摄像头,笔记本内置摄像头为0，USB外接摄像头为1或2，深度图或RGB

cap = cv2.VideoCapture(4)  # 或者使用正确的设备索引
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
cap.set(3,1280)
cap.set(4,720)
font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)
# num = 0
while True:
    start = time.time()
    ret, frame = cap.read()
    # operations on the frame come here

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)    #修改成自己需求格式的Aruco二维码图片
    parameters = aruco.DetectorParameters()

    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          aruco_dict,
                                                          parameters=parameters)
    if ids is not None:
        # 0.1是arucco码的边长
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, mtx, dist)
        (rvec - tvec).any()  # get rid of that nasty numpy value array error

        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.1)
            aruco.drawDetectedMarkers(frame, corners)

            # 将 rvec 转换为旋转矩阵
            rotation = R.from_rotvec(rvec[i, 0, :])  # 将 rvec 的形状转换为 (3,)
            euler_angles = rotation.as_euler('xyz', degrees=False)

            # 显示ID，rvec, tvec, 欧拉角 (弧度制)
            cv2.putText(frame, "Id:" + str(ids[i]), (0, 40 + i * 20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Euler: " + str(euler_angles), (0, 60 + i * 20), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            result = [tvec[i, :, :][0][0],tvec[i, :, :][0][1],tvec[i, :, :][0][2],euler_angles[0],euler_angles[1],euler_angles[2]]
            rounded_list = [round(num, 5) for num in result]
            print(rounded_list)
            time.sleep(0.3)
            
    else:
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    end = time.time()
    # 计算帧率并显示
    cv2.putText(frame, "rate: " + str(1 / (end - start)), (10, 120), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == 27:  # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break
    if key == ord(' '):  # 按空格键保存
        #        num = num + 1
        #        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)