#!/usr/bin/env python3
"""D405 ChArUco pose TF for easy_handeye2; ROS 2 Humble, OpenCV ArUco."""
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster


def quat_from_rotation(r):
    """Return normalized xyzw quaternion from a 3x3 rotation matrix."""
    q = np.empty(4, dtype=np.float64)
    t = float(np.trace(r))
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q[:] = [(r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(0.0, 1.0 + r[i, i] - r[j, j] - r[k, k]))
        if s < 1e-12:
            raise ValueError('degenerate rotation')
        q[i] = 0.25 * s
        q[j] = (r[i, j] + r[j, i]) / s
        q[k] = (r[i, k] + r[k, i]) / s
        q[3] = (r[k, j] - r[j, k]) / s
    return q / np.linalg.norm(q)


class CharucoTF(Node):
    def __init__(self):
        super().__init__('charuco_tf_node')
        for name, default in (
            ('image_topic', '/camera/camera/color/image_raw'),
            ('camera_info_topic', '/camera/camera/color/camera_info'),
            ('board_frame', 'charuco_board'),
            ('squares_x', 8), ('squares_y', 6),
            ('square_length_m', 0.020), ('marker_length_m', 0.015),
            ('min_corners', 8), ('max_reproj_error_px', 2.0),
            ('publish_debug_image', True)):
            self.declare_parameter(name, default)
        self.board_frame = self.get_parameter('board_frame').value
        cols = int(self.get_parameter('squares_x').value)
        rows = int(self.get_parameter('squares_y').value)
        sl = float(self.get_parameter('square_length_m').value)
        ml = float(self.get_parameter('marker_length_m').value)
        if not (cols >= 2 and rows >= 2 and 0 < ml < sl):
            raise ValueError('Invalid ChArUco board geometry')
        if not hasattr(cv2, 'aruco'):
            raise RuntimeError('Current ROS Python OpenCV lacks cv2.aruco (opencv-contrib required).')
        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        # IMPORTANT: the printed target was generated with the modern CharucoBoard API.
        # OpenCV changed the ChArUco pattern convention in 4.6 for boards with an
        # even number of rows. Your board has 6 rows, so OpenCV 4.5.x's legacy
        # CharucoBoard_create geometry is NOT a safe match for the printed board.
        if hasattr(aruco, 'CharucoBoard'):
            self.board = aruco.CharucoBoard((cols, rows), sl, ml, self.dictionary)
            if hasattr(self.board, 'setLegacyPattern'):
                self.board.setLegacyPattern(False)
        else:
            raise RuntimeError(
                'This printed 8x6 ChArUco target requires the modern OpenCV ChArUco '
                'pattern API (OpenCV >= 4.6; >= 4.8 recommended). The current '
                f'OpenCV is {cv2.__version__}. Do not use CharucoBoard_create with '
                'this already-printed board because 6 is an even row count and the '
                'legacy pattern convention differs.'
            )
        self.detector = aruco.CharucoDetector(self.board) if hasattr(aruco, 'CharucoDetector') else None
        self.obj_points = (self.board.getChessboardCorners() if hasattr(self.board, 'getChessboardCorners')
                           else self.board.chessboardCorners)
        self.obj_points = np.asarray(self.obj_points, np.float64).reshape(-1, 3)
        self.min_corners = int(self.get_parameter('min_corners').value)
        self.max_rmse = float(self.get_parameter('max_reproj_error_px').value)
        self.bridge = CvBridge()
        self.tf_pub = TransformBroadcaster(self)
        self.K = self.D = None
        self.info_frame = None
        self.info_size = None
        info_topic = self.get_parameter('camera_info_topic').value
        image_topic = self.get_parameter('image_topic').value
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.on_info, 10)
        self.image_sub = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.debug_pub = (self.create_publisher(Image, '~/debug_image', 10)
                          if self.get_parameter('publish_debug_image').value else None)
        self.frame_count = 0
        self.get_logger().info(f'Waiting for {info_topic} and {image_topic}; target TF: optical frame -> {self.board_frame}')

    def on_info(self, msg):
        if not msg.header.frame_id or msg.width <= 0 or msg.height <= 0:
            return
        K = np.array(msg.k, np.float64).reshape(3, 3)
        D = np.array(msg.d, np.float64).reshape(-1, 1)
        if msg.distortion_model not in ('plumb_bob', 'rational_polynomial'):
            self.get_logger().warning(f'Unsupported distortion model: {msg.distortion_model}', throttle_duration_sec=5.0)
            self.K = None
            return
        if not np.isfinite(K).all() or not np.isfinite(D).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            self.K = None
            return
        self.K, self.D = K, D
        self.info_frame = msg.header.frame_id.lstrip('/')
        self.info_size = (msg.width, msg.height)

    def on_image(self, msg):
        if self.K is None:
            self.get_logger().warning('Waiting for valid camera_info', throttle_duration_sec=5.0)
            return
        if (msg.width, msg.height) != self.info_size or msg.header.frame_id.lstrip('/') != self.info_frame:
            self.get_logger().warning('Image and camera_info frame or resolution mismatch; no TF', throttle_duration_sec=5.0)
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.detector is not None:
                corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
                corners = ids = None
                if marker_ids is not None:
                    _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, self.board, cameraMatrix=self.K, distCoeffs=self.D)
            count = 0 if ids is None else len(ids)
            status = f'corners={count}  NO TF'
            if ids is not None and count >= self.min_corners:
                selected_ids = np.asarray(ids, np.int32).reshape(-1)
                if np.any(selected_ids < 0) or np.any(selected_ids >= len(self.obj_points)):
                    raise ValueError('Unexpected ChArUco corner ID')
                object_points = self.obj_points[selected_ids]
                image_points = np.asarray(corners, np.float64).reshape(-1, 2)
                ok, rvec, tvec = cv2.solvePnP(object_points, image_points, self.K, self.D,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.K, self.D)
                    rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
                    z = float(tvec[2, 0])
                    if np.isfinite(rmse) and rmse <= self.max_rmse and z > 0.0:
                        rotation, _ = cv2.Rodrigues(rvec)
                        q = quat_from_rotation(rotation)
                        tf = TransformStamped()
                        tf.header = msg.header   # Image exposure timestamp, not now().
                        tf.header.frame_id = self.info_frame
                        tf.child_frame_id = self.board_frame
                        tf.transform.translation.x = float(tvec[0, 0])
                        tf.transform.translation.y = float(tvec[1, 0])
                        tf.transform.translation.z = z
                        tf.transform.rotation.x = float(q[0])
                        tf.transform.rotation.y = float(q[1])
                        tf.transform.rotation.z = float(q[2])
                        tf.transform.rotation.w = float(q[3])
                        self.tf_pub.sendTransform(tf)
                        cv2.drawFrameAxes(image, self.K, self.D, rvec, tvec, 0.03)
                        status = f'corners={count}  rmse={rmse:.2f}px  z={z:.3f}m  TF OK'
                    else:
                        status = f'corners={count}  rmse={rmse:.2f}px  NO TF'
            if ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(status)
            if self.debug_pub:
                cv2.putText(image, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.60, (0, 255, 0) if status.endswith('TF OK') else (0, 0, 255), 2)
                annotated = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                annotated.header = msg.header
                self.debug_pub.publish(annotated)
        except Exception as exc:
            self.get_logger().error(f'Frame rejected; no TF: {exc}', throttle_duration_sec=3.0)


def main():
    rclpy.init()
    node = None
    try:
        node = CharucoTF()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
