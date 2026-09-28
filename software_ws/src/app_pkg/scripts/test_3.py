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
def resample_curve(points, n):
    """
    将曲线均匀重采样为n个点
    :param points: 原始曲线点 (N,2)数组
    :param n: 目标点数
    :return: 均匀重采样后的点 (n,2)数组
    """
    if len(points) < 2:
        return points
    
    # 计算累积弧长
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.cumsum(segment_lengths)
    total_length = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    
    # 生成等距参数
    if total_length == 0:  # 所有点重合的情况
        return np.tile(points[0], (n, 1))
    
    target_lengths = np.linspace(0, total_length, n)
    
    # 线性插值
    resampled = []
    j = 0
    for length in target_lengths:
        # 找到当前长度所在的线段
        while j < len(cumulative_length) - 1 and cumulative_length[j] < length:
            j += 1
        
        if j == 0:
            t = length / cumulative_length[0] if cumulative_length[0] > 0 else 0
        else:
            t = (length - cumulative_length[j-1]) / (cumulative_length[j] - cumulative_length[j-1])
        
        # 插值
        interpolated_point = points[j] + t * (points[j+1] - points[j])
        resampled.append(interpolated_point)
    
    return np.array(resampled)

# 修改后的主程序
if __name__ == "__main__":
    test_cases = [
        np.array([[0,0], [1,2], [3,1], [5,3]]),
        np.array([[0,0], [0,0], [1,1], [1,1], [2,0]]),
        np.array([[0,0], [1,0], [1,1]]),
        np.array([[0,0], [1,1], [2,2], [3,1], [4,0]])
    ]
    
    for i, original_points in enumerate(test_cases):
        plt.figure(figsize=(12, 5))
        
        # 原始折线
        plt.subplot(1, 2, 1)
        plt.plot(original_points[:,0], original_points[:,1], 'ro--', markersize=8, label='Original')
        plt.title(f"Original (Case {i+1})")
        plt.axis('equal')
        plt.grid(True)
        
        # 平滑并重采样
        plt.subplot(1, 2, 2)
        smoothed = smooth_with_rounded_corners(original_points, radius=0.4)
        uniform_resampled = resample_curve(smoothed, n=100)  # 重采样为20个点
        
        plt.plot(smoothed[:,0], smoothed[:,1], 'b-', alpha=0.3, label='Smoothed')
        plt.plot(uniform_resampled[:,0], uniform_resampled[:,1], 'go-', markersize=6, label='Uniform Resampled (20 points)')
        plt.title(f"Resampled (Case {i+1})")
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()