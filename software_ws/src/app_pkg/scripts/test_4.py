import numpy as np
import matplotlib.pyplot as plt

def smooth_with_rounded_corners(points, radius=0.3, resolution=20):
    """
    完全稳定的折线圆角平滑方案
    特点：
    1. 不依赖scipy，仅使用numpy
    2. 自动处理重复点、共线点等边界情况
    3. 自适应半径控制
    """
    points = np.asarray(points, dtype=np.float64)
    
    # 预处理：移除重复点和过于接近的点
    unique_points = []
    for i, p in enumerate(points):
        if i == 0:
            unique_points.append(p)
        else:
            dist = np.linalg.norm(p - unique_points[-1])
            if dist > 1e-6:  # 距离阈值
                unique_points.append(p)
    
    points = np.array(unique_points)
    if len(points) < 3:
        return points
    
    # 主处理流程
    smoothed_points = [points[0]]
    n = len(points)
    
    for i in range(1, n-1):
        prev = points[i-1]
        curr = points[i]
        next_p = points[i+1]
        
        # 计算线段向量
        vec1 = curr - prev
        vec2 = next_p - curr
        
        # 计算线段长度
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        
        # 跳过太短的线段
        if len1 < 1e-6 or len2 < 1e-6:
            continue
            
        # 归一化向量
        vec1 = vec1 / len1
        vec2 = vec2 / len2
        
        # 计算实际可用半径
        max_r = min(len1, len2) * 0.4  # 保守系数
        actual_r = min(radius, max_r)
        
        # 计算控制点
        angle = np.arccos(np.dot(vec1, vec2))
        if angle < 0.1 or angle > np.pi - 0.1:  # 近似直线
            smoothed_points.append(curr)
            continue
            
        # 计算圆角起始点和结束点
        start = curr - vec1 * actual_r
        end = curr + vec2 * actual_r
        
        # 生成贝塞尔曲线点
        for t in np.linspace(0, 1, resolution):
            # 二次贝塞尔曲线
            point = (1-t)**2 * start + 2*(1-t)*t * curr + t**2 * end
            smoothed_points.append(point)
    
    # 添加最后一个点
    if n > 1:
        smoothed_points.append(points[-1])
    
    return np.array(smoothed_points)

# 示例使用
if __name__ == "__main__":
    # 测试数据（包含各种边界情况）
    test_cases = [
        # 普通折线
        np.array([[0,0], [1,2], [3,1], [5,3]]),
        
        # 包含重复点
        np.array([[0,0], [0,0], [1,1], [1,1], [2,0]]),
        
        # 锐角折线
        np.array([[0,0], [1,0], [1,1]]),
        
        # 共线点
        np.array([[0,0], [1,1], [2,2], [3,1], [4,0]])
    ]
    
    for i, original_points in enumerate(test_cases):
        plt.figure(figsize=(10, 5))
        
        # 原始折线
        plt.plot(original_points[:,0], original_points[:,1], 'ro--', 
                 markersize=8, linewidth=1, label='Original')
        
        # 平滑结果
        for radius in [0.2, 0.4, 0.6]:
            smoothed = smooth_with_rounded_corners(original_points, radius)
            print("结果",smoothed)
            plt.plot(smoothed[:,0], smoothed[:,1], '-', 
                     linewidth=2, label=f'Radius={radius}')
        
        plt.title(f"Test Case {i+1}")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()