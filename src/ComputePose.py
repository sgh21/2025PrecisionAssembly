from ultralytics import YOLO
import torch
import numpy as np
import cv2
import os
import sys
from typing import List, Tuple
from sklearn.cluster import KMeans

current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))

from utils.SegmentResult import Const, SegmentResult


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO("../weights/yolov8-hole.pt")  # Load a trained model


def filter_keyhole_cicle(seg:SegmentResult) -> 'List[List[int,int]]':
    '''对于keyhole类(齿轮内轮廓)，额外将bbox纵向2/3以下部分的点删去
    不是所有图像都沿纵向保留2/3，考虑引入角度参数 -> 弃用本函数
    '''
    valid_points = []
    if seg.class_name == "keyhole":
        valid_points = seg.edge_point_i # 获取边缘点列表
        # 对于keyhole类，排除bbox纵向2/3以下部分的点
        y_threshold = seg.xyxy_i[1] + (seg.xyxy_i[3] - seg.xyxy_i[1]) * 2 / 3
        valid_points = [p for p in valid_points if p[0] < y_threshold]
    return valid_points

def filter_gear_circle(gear_pos:'Tuple[int,int,float]', seg:SegmentResult) -> SegmentResult:
    '''
    对于gear类，确定齿轮位置和内轮廓后，将外轮廓边界中的内轮廓点删去
    '''
    if seg.class_name != "gear":
        return seg
    # 获取边缘点列表
    edge_points = seg.edge_point_i
    edge_points_relative = edge_points - np.array([gear_pos[1], gear_pos[0]])  # 转换为相对坐标系
    # 计算半径
    radius_inner = gear_pos[2]
    radius = np.linalg.norm(edge_points_relative, axis=1)
    mask = radius > radius_inner * 2  # 保留半径大于内轮廓半径n倍的点
    seg.edge_point_i = seg.edge_point_i[mask]
    # seg.edge_point_l = seg.edge_point_i[mask] # 数组大小不匹配，不再进行过滤

def locate_circle(points:'List[List[int,int]]')-> Tuple[float, float, float]:
    '''
    使用代数最小二乘法拟合圆心和半径
    points: list of [y, x] points, 原图坐标系
    返回: (center_x, center_y, radius), 原图坐标系
    '''
    if len(points) < 3:
        return None, None, None
    points = np.array(points, dtype=float)
    x = points[:, 1]
    y = points[:, 0]
    n = len(x)
    # 计算质心
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # 中心化坐标
    u = x - x_mean
    v = y - y_mean
    # 构建系数矩阵
    Suu = np.sum(u * u)
    Suv = np.sum(u * v)
    Svv = np.sum(v * v)
    Suuu = np.sum(u * u * u)
    Suvv = np.sum(u * v * v)
    Svvv = np.sum(v * v * v)
    Suuv = np.sum(u * u * v)
    # 求解线性系统
    A = np.array([[Suu, Suv],
                  [Suv, Svv]])
    b = np.array([0.5 * (Suuu + Suvv),
                  0.5 * (Svvv + Suuv)])
    uc, vc = np.linalg.solve(A, b)
    # 转换回原坐标系
    center_x = uc + x_mean
    center_y = vc + y_mean
    # 计算半径
    radius = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / n)
    return center_x, center_y, radius

def involute_equation(t, rb, theta0):
    """
    渐开线参数方程
    t: 参数（展开角）
    rb: 基圆半径
    theta0: 初始角度偏移
    """
    x = rb * (np.cos(t + theta0) + t * np.sin(t + theta0))
    y = rb * (np.sin(t + theta0) - t * np.cos(t + theta0))
    return x, y

from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

def fit_single_involute(points, rb_initial, theta0_initial=0):
    """
    拟合单条渐开线
    points: np.array, shape (N, 2), 渐开线上的点坐标 [x, y]
    rb_initial: 基圆半径初值
    theta0_initial: 初始角度偏移初值
    """
    points = np.array(points)
    x_data, y_data = points[:, 0], points[:, 1]
    
    def residual_function(params):
        rb, theta0, t_offset = params
        residuals = []
        
        for i, (x_obs, y_obs) in enumerate(points):
            # 为每个观测点找到最佳的参数t
            def point_distance(t):
                x_pred, y_pred = involute_equation(t + t_offset, rb, theta0)
                return (x_pred - x_obs)**2 + (y_pred - y_obs)**2
            
            # 初始t值估计
            r_obs = np.sqrt(x_obs**2 + y_obs**2)
            t_init = max(0, np.sqrt((r_obs/rb)**2 - 1))
            
            # 优化找到最佳t
            result = minimize(point_distance, t_init, bounds=[(0, 5)])
            if result.success:
                t_opt = result.x[0]
                x_pred, y_pred = involute_equation(t_opt + t_offset, rb, theta0)
                residuals.append(np.sqrt((x_pred - x_obs)**2 + (y_pred - y_obs)**2))
            else:
                residuals.append(1000)  # 大的残差值
        
        return residuals
    
    # 参数初值：[rb, theta0, t_offset]
    initial_params = [rb_initial, theta0_initial, 0.0]
    
    # 参数边界
    bounds = ([rb_initial*0.8, -np.pi, -1], 
             [rb_initial*1.2, np.pi, 1])
    
    try:
        result = least_squares(residual_function, initial_params, 
                             bounds=bounds, max_nfev=1000)
        
        if result.success:
            rb_fit, theta0_fit, t_offset_fit = result.x
            rms_error = np.sqrt(np.mean(np.array(result.fun)**2))
            
            return {
                'rb': rb_fit,
                'theta0': theta0_fit,
                't_offset': t_offset_fit,
                'rms_error': rms_error,
                'success': True,
                'params': result.x
            }
    except Exception as e:
        print(f"拟合失败: {e}")
    
    return {'success': False}

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def merge_involute_segments(segments):
    """
    合并属于同一条渐开线的多个片段
    """
    if len(segments) <= 1:
        return segments[0] if segments else np.array([])
    
    # 将所有点合并
    all_points = np.vstack(segments)
    
    # 按距离原点的距离排序（渐开线特性：随参数t增加，半径增加）
    distances = np.sqrt(all_points[:, 0]**2 + all_points[:, 1]**2)
    sorted_indices = np.argsort(distances)
    
    return all_points[sorted_indices]

def cluster_involutes_dbscan(involutes, eps=0.1, min_samples=5):
    """
    使用DBSCAN对渐开线片段进行聚类
    """
    if len(involutes) <= 1:
        return involutes
    
    # 提取每个片段的特征
    features = []
    segment_indices = []
    
    for i, inv in enumerate(involutes):
        if len(inv) > 0:
            # 特征：中心角度、平均半径、角度范围
            center_x = np.mean(inv[:, 0])
            center_y = np.mean(inv[:, 1])
            center_angle = np.arctan2(center_y, center_x)
            center_angle = (center_angle + 2*np.pi) % (2*np.pi)
            
            avg_radius = np.mean(np.sqrt(inv[:, 0]**2 + inv[:, 1]**2))
            
            angles = np.arctan2(inv[:, 1], inv[:, 0])
            angles = (angles + 2*np.pi) % (2*np.pi)
            angle_range = np.max(angles) - np.min(angles)
            if angle_range > np.pi:  # 处理跨越0°的情况
                angle_range = 2*np.pi - angle_range
            
            features.append([center_angle, avg_radius, angle_range])
            segment_indices.append(i)
    
    if len(features) < 2:
        return involutes
    
    # 标准化特征
    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)
    labels = clustering.labels_
    
    # 合并同一聚类的片段
    merged_involutes = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:  # 噪声点，单独处理
            noise_indices = [segment_indices[i] for i, l in enumerate(labels) if l == -1]
            for idx in noise_indices:
                merged_involutes.append(involutes[idx])
        else:
            # 合并同一聚类的片段
            cluster_indices = [segment_indices[i] for i, l in enumerate(labels) if l == label]
            segments_to_merge = [involutes[i] for i in cluster_indices]
            merged_points = merge_involute_segments(segments_to_merge)
            if len(merged_points) > 0:
                merged_involutes.append(merged_points)
    
    print(f"DBSCAN聚类: {len(involutes)} -> {len(merged_involutes)} 个渐开线")
    return merged_involutes

def visualize_involute_fitting(original_involutes, fitted_involutes, rb_initial):
    """可视化拟合结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 绘制原始点
    colors = plt.cm.tab10(np.linspace(0, 1, len(original_involutes)))
    
    for i, inv_points in enumerate(original_involutes):
        if len(inv_points) > 0:
            ax1.scatter(inv_points[:, 0], inv_points[:, 1], 
                       c=[colors[i]], s=20, alpha=0.7, label=f'origin {i}')
    
    # 绘制拟合曲线
    for fit_data in fitted_involutes:
        params = fit_data['fit_params']
        rb, theta0, t_offset = params['rb'], params['theta0'], params['t_offset']
        
        # 生成拟合曲线
        t_range = np.linspace(0, 3, 200)
        x_fit, y_fit = involute_equation(t_range + t_offset, rb, theta0)
        
        ax2.plot(x_fit, y_fit, '--', linewidth=2, 
                label=f'Fitted involutes {fit_data["index"]} (rb={rb:.1f})')
        
        # 绘制原始点
        points = fit_data['points']
        ax2.scatter(points[:, 0], points[:, 1], s=30, alpha=0.8)
    
    # 绘制基圆
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_circle = rb_initial * np.cos(theta_circle)
    y_circle = rb_initial * np.sin(theta_circle)
    ax1.plot(x_circle, y_circle, 'k--', alpha=0.5, label='base circle')
    ax2.plot(x_circle, y_circle, 'k--', alpha=0.5, label='base circle')
    
    ax1.set_title('sampled points')
    ax2.set_title('fitted involutes')
    ax1.legend()
    ax2.legend()
    ax1.axis('equal')
    ax2.axis('equal')
    ax1.grid(True)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def calculate_angle_with_involutes(theta, rho, edge_points_relative):
    size = len(theta)
    tooth_height = (np.max(rho) - np.min(rho))          # 齿高
    radius_pitch = (np.max(rho) + np.min(rho)) / 2      # 分度圆半径
    radius_base = radius_pitch * np.cos(Const.PRESSURE_ANGLE)    # 基圆半径
    involutes = []      # 渐开线列表
    stt_all_idx = -1    # 第一个半径极小值点索引
    stt_idx = 0        # 当前渐开线起始点索引
    idx = 0
    state = 0  # 状态机：0-任意， 1-寻找极小值点，2-寻找极大值点
    while idx < size + stt_all_idx:
        idx += 1
        ii = (idx-2) % size  # 处理循环
        i = (idx-1) % size
        j = idx % size
        k = (idx+1) % size
        kk = (idx+2) % size
        
        delta_theta = theta[j] - theta[stt_idx]
        if delta_theta > 0.5 * 2*np.pi/Const.GEAR_Z:    # 超过半齿，舍弃
            stt_idx = j  # 更新起始点索引
            state = 0
            continue
        if state!=2 and rho[ii]>rho[i] and rho[i]>rho[j] and rho[j]<rho[k] and rho[k]<rho[kk]:  # 找到极小值点
            if stt_all_idx == -1:
                stt_all_idx = j
                stt_idx = j
            
            if j > stt_idx:  # 如果临时存储的渐开线点不为空，检查并保存
                # print(f'min rho locate at:{rho[j]:.2f}, {theta[j]*180/np.pi:.1f}°')
                print(f'min idx: {j}')
                # 终止条件：片段角度范围至少1/5齿，终止点至少低于rp-1/3h
                # if delta_theta > 0.2 * 2*np.pi/Const.GEAR_Z and rho[j]<radius_pitch-1/3*tooth_height:
                rhos = rho[stt_idx:j]
                if np.max(rhos) - np.min(rhos) > tooth_height / 2:  # 如果跨过至少1/2齿高
                    # 保留rp-1/4h~rp+1/4h部分的点
                    mask = np.zeros(size, dtype=bool)
                    if j > stt_idx:
                        mask[stt_idx:j] = (rhos > radius_base - tooth_height / 4) & (rhos < radius_base + tooth_height / 4)
                    else:
                        mask[stt_idx:] = (rhos > radius_base - tooth_height / 4) & (rhos < radius_base + tooth_height / 4)
                        mask[:j] = (rho[:j] > radius_base - tooth_height / 4) & (rho[:j] < radius_base + tooth_height / 4)
                    involutes.append(edge_points_relative[mask])
                    
                # 更新起始点条件：起始点低于rp-1/3h
                if rho[j] < radius_pitch - 0.4 * tooth_height:
                    stt_idx = j  # 更新起始点索引
                    state = 2
        if state!=1 and rho[ii]<rho[i] and rho[i]<rho[j] and rho[j]>rho[k] and rho[k]>rho[kk]:  # 找到极大值点
            if j > stt_idx:  # 如果临时存储的渐开线点不为空，检查并保存
                print(f'max idx: {j}')
                delta_theta = theta[j] - theta[stt_idx]
                # 终止条件：片段角度范围至少1/5齿，终止点至少高于rp+1/3h
                # if delta_theta > 0.2 * 2*np.pi/Const.GEAR_Z and rho[j]>radius_pitch+1/3*tooth_height:
                rhos = rho[stt_idx:j]
                if max(rhos) - min(rhos) > tooth_height / 2:  # 如果跨过至少1/2齿高
                    # 保留rp-1/4h~rp+1/4h部分的点
                    mask = np.zeros(size, dtype=bool)
                    if j > stt_idx:
                        mask[stt_idx:j] = (rhos > radius_base - tooth_height/3) & (rhos < radius_base + tooth_height /3)
                    else:
                        mask[stt_idx:] = (rhos > radius_base - tooth_height/3) & (rhos < radius_base + tooth_height /3)
                        mask[:j] = (rho[:j] > radius_base - tooth_height/3) & (rho[:j] < radius_base + tooth_height /3)
                    involutes.append(edge_points_relative[mask])
                if rho[j] > radius_pitch + 0.4 * tooth_height:
                    stt_idx = j  # 更新起始点索引
                    state = 1
            
    print(f"Detected {len(involutes)} involutes.")
    # img = seg.img_i.copy()  # 检查渐开线分割
    # for idx, inv in enumerate(involutes):
    #     for p in inv:
    #         x = int(cx + p[1])
    #         y = int(cy + p[0])
    #         if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
    #             img = cv2.circle(img, (x, y), radius=1, color=(255, min(20*i,255), 0), thickness=3)
    # img = cv2.resize(img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
    # cv2.imshow("Invulutes", img)
    # cv2.waitKey(0)
    # involutes = cluster_involutes_dbscan(involutes, eps=0.1, min_samples=3)  # 聚类渐开线片段
    fitted_involutes = []  # 存储拟合结果
    for idx, inv in enumerate(involutes):
        if len(inv) < 8:
            continue
        theta_ini = np.arctan2(np.mean(inv[:, 1]),np.mean(inv[:, 0]))  # 初始角度
        fit_result = fit_single_involute(inv, rb_initial=radius_base, theta0_initial=theta_ini)
        if fit_result['success']:
            fitted_involutes.append({
                'index': i,
                'points': inv,
                'fit_params': fit_result,
                'tooth_angle': theta_ini  # 该齿的角度位置
            })
    visualize_involute_fitting(involutes, fitted_involutes, radius_base)

def fps_downsample_kmeans(points, num_samples):
    """
    使用KMeans聚类实现类似FPS的效果
    """
    points = np.array(points)
    if len(points) <= num_samples:
        return points
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init=10)
    kmeans.fit(points)
    
    # 返回聚类中心
    return kmeans.cluster_centers_

def remove_outliers_mad(data, threshold=2.0):
    """
    使用中位数绝对偏差(MAD)方法去除离群值
    更robust，不受极端值影响
    """
    if len(data) < 3:
        return data
    
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        return data
    
    modified_z_scores = 0.6745 * (data - median) / mad
    mask = np.abs(modified_z_scores) < threshold
    return mask


def calculate_gear_angle(gear_pos:Tuple[int,int,float], seg:SegmentResult) -> float:
    '''计算齿轮的旋转角度
    返回：范围 [0, 2*pi/z]
    '''
    if seg.class_name != "gear":
        return None
    mask_edge = True  # 是否使用mask边界点
    if mask_edge:
        edge_points = seg.mask_edge_point_o/seg.scale_io
    else:
        edge_points = seg.edge_point_i
    cx, cy = gear_pos[:2]  # 齿轮中心位置
    edge_points_relative = edge_points - np.array([cy, cx])  # 转换为相对坐标系
    # 对边缘点进行下采样，增大采样点间距，计算齿顶角度均值时能够利用多个齿的齿顶
    # edge_points_relative = fps_downsample_kmeans(edge_points_relative, num_samples=len(edge_points_relative)//10)
    u, v = edge_points_relative[:, 1], edge_points_relative[:, 0]
    rho, theta = np.sqrt(u**2+v**2), np.arctan2(v, u)   # 极坐标转换
    theta = np.mod(theta, 2 * np.pi)                    # 确保角度在 [0, 2*pi] 范围内
    # print(f"theta range: {np.min(theta)} to {np.max(theta)}")
    # print(f"rho range: {np.min(rho)} to {np.max(rho)}")
    tooth_height = (np.max(rho) - np.min(rho))          # 齿高
    radius_pitch = (np.max(rho) + np.min(rho)) / 2      # 分度圆半径
    radius_base = radius_pitch * np.cos(Const.PRESSURE_ANGLE)    # 基圆半径
    # 按照角度排序
    sorted_indices = np.argsort(theta)
    theta = theta[sorted_indices]
    rho = rho[sorted_indices]
    edge_points_relative = edge_points_relative[sorted_indices]
    size = len(theta)    
    
    # 方法一：按渐开线切片，按角度顺序，到极大值或极小值时切片，若跨过至少1/2齿高则保留rp-1/4h~rp+1/4h部分的点
    # calculate_angle_with_involutes(theta, rho, edge_points_relative)  # 切片效果不佳

    # 方法二：暴力搜索rho的top-k点的theta，归一化到单齿角度范围后取均值 # TODO：考虑使用pitch上的边界点，先聚为两类分别取均值，再对两个角度取均值（无偏）
    # 找到最大rho所在theta
    tooth_range = 2 * np.pi / Const.GEAR_Z  # 单齿角度范围
    theta_peek = np.mod(theta[np.argmax(rho)]+1/2*tooth_range, tooth_range)  # 最大rho对应的theta
    # 根据齿轮齿数，将theta切分成z个部分
    theta_split = []
    rho_split = []
    for i in range(Const.GEAR_Z):
        start_angle = (theta_peek + i * tooth_range) % (2 * np.pi)
        end_angle = (theta_peek + (i + 1) * tooth_range) % (2 * np.pi)
        if start_angle < end_angle:
            mask = (theta >= start_angle) & (theta < end_angle)
        else:# 处理跨越0度的情况
            mask = (theta >= start_angle) | (theta < end_angle)
        # 提取该区间内的点
        theta_split.append(theta[mask])
        rho_split.append(rho[mask])
    
    top_rho_theta = []  # 存储每个区间的top-k theta
    top_rho_edge_points = []  # 存储每个区间的top-k边缘点
    k = 1  # 每个区间取前k个rho极值点
    for theta, rho in zip(theta_split, rho_split):
        k = min(len(theta), k)
        if k == 0:
            continue
        top_k_indices = np.argsort(rho)[-k:]  # 获取rho的top-k点索引
        top_rho_theta.append(theta[top_k_indices])  # 获取对应的theta值
        for idx in top_k_indices:
            cx, cy = gear_pos[:2]
            # 获取对应的边缘点
            r = rho[idx]
            t = theta[idx]
            point = [int(cy + r * np.sin(t)), int(cx + r * np.cos(t))]
            top_rho_edge_points.append(point)

    regularized_theta = np.mod(np.array(top_rho_theta).flatten(), tooth_range)  # 确保角度在 [0, 2*pi/z] 范围内
    mask = remove_outliers_mad(regularized_theta, threshold=2.0)  # 使用MAD方法去除离群值
    regularized_theta = regularized_theta[mask]  # 过滤离群值
    # print(f"Regularized theta range: {np.min(regularized_theta)} to {np.max(regularized_theta)}")
    # print(regularized_theta)
    gear_angle_mean = np.mean(regularized_theta)  # 计算平均角度
    gear_angle_median = np.median(regularized_theta)  # 计算中位数角度
    # print(f"Mean: {gear_angle_mean}, Median: {gear_angle_median}")

    # img = seg.img_i.copy()  # 检查齿轮角度计算
    # for p in edge_points_relative:
    #     x = int(cx + p[1])
    #     y = int(cy + p[0])
    #     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
    #         img = cv2.circle(img, (x, y), radius=1, color=(255, 0, 0), thickness=5)
    # img = cv2.line (img, (cx, cy), (int(cx + 500 * np.cos(gear_angle_mean)), int(cy + 500 * np.sin(gear_angle_mean))), (0, 255, 255), 10)
    # img = cv2.line (img, (cx, cy), (int(cx + 500 * np.cos(gear_angle_median)), int(cy + 500 * np.sin(gear_angle_median))), (0, 0, 255), 10)
    # img = cv2.resize(img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
    # cv2.imshow("Gear Angle", img)

    return gear_angle_mean, top_rho_edge_points



def locate_all_segment(seg_list:List[SegmentResult], img=None, debug=False) -> Tuple[Tuple[int,int,float],List[Tuple[int,int,float]],float]:
    '''综合处理所有分割对象，输出：齿轮和6个孔的位置+半径，以及齿轮的旋转角度
    '''
    gear_pos = (0, 0)   # 齿轮位置，原图坐标系
    hole_pos_list = []  # 孔洞位置列表，原图坐标系
    gear_angle = None   # 范围 [0, 2*pi/z]
    gear_list = []
    keyhole_list = []
    hole_list = []
    for seg in seg_list:
        if seg.class_name == "gear":
            gear_list.append(seg)
        elif seg.class_name == "hole":
            hole_list.append(seg)
        elif seg.class_name == "keyhole":
            keyhole_list.append(seg)
            
    '''1. 计算齿轮的位置'''
    if len(keyhole_list) != 1:
        print(f"WARNING: Keyhole Detection Failed. Detected Number:{len(keyhole_list)}")
    for seg in keyhole_list:
        # edge_points = filter_keyhole_cicle(seg)
        edge_points = seg.edge_point_i    # 使用边缘点（不过滤），直接进行圆拟合
        cx, cy, r = locate_circle(edge_points)

        # if debug and img is not None: # 绘制keyhole边界
        #     for p in edge_points:
        #         img = cv2.circle(img, tuple(p[::-1]), radius=1, color=(0, 255, 0), thickness=2)
        #     print(f"filtered keyhole points: {len(edge_points)}")
        #     img = cv2.circle(img, (int(cx), int(cy)), radius=int(r), color=(0, 255, 0), thickness=2)
        #     img = cv2.resize(img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
        #     cv2.imshow("Keyhole Circle", img)
        #     cv2.waitKey(0)
        if cx is not None and cy is not None and r is not None:
            gear_pos= (int(cx), int(cy),r)
    
    # # 使用霍夫圆变换更精确地定位齿轮位置
    for seg in keyhole_list:
        limg = seg.local_img_i.copy()
        gray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 2)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=100, param2=30, minRadius=50, maxRadius=200)
        print(f"Detected circles: {len(circles[0]) if circles is not None else 0}")
        if circles is not None:
            error = []
            for circle in circles[0, :]:
                cx, cy, r = circle
                ci = np.array([seg.xyxy_i[1], seg.xyxy_i[0]])+np.array([cx, cy])  # 转换为原图坐标系
                if np.linalg.norm(ci - np.array(gear_pos[:2])) > 30:
                    error.append(1e6)  # 距离过远，认为是错误检测
                    continue
                error.append(10*np.linalg.norm(ci - np.array(gear_pos[:2]))+abs(r+10 - gear_pos[2]))
                if debug:
                    limg = cv2.circle(limg, (int(cx), int(cy)), radius=int(r), color=(0, 255, 0), thickness=2)
                    print(f"Detected circle: center=({cx}, {cy}), radius={r}") 
            best_circle = circles[0, np.argmin(error)]
            gear_pos = (int(seg.xyxy_i[0] + best_circle[0]), int(seg.xyxy_i[1] + best_circle[1]), best_circle[2])

            # if limg is not None and debug:
            #     cv2.circle(limg, (int(best_circle[0]), int(best_circle[1])), radius=int(best_circle[2]), color=(255, 255, 0), thickness=3)
            #     cv2.imshow("Hough Circle Detection", limg)
            #     cv2.waitKey(0)
    
    '''2. 计算齿轮的角度'''
    if len(gear_list) != 1:
        print(f"WARNING: Gear Detection Failed. Detected Number:{len(gear_list)}")
    for seg in gear_list:
        filter_gear_circle(gear_pos, seg)  # 过滤掉齿轮内轮廓点
        gear_angle, top_rho_points = calculate_gear_angle(gear_pos, seg)
    if debug and img is not None:
        for p in top_rho_points:
            img = cv2.circle(img, tuple(p[::-1]), radius=5, color=(255, 0, 0), thickness=8)
        # img = cv2.resize(img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
        # cv2.imshow("Gear Angle Detection", img)
        # cv2.waitKey(0)
    
    '''3. 计算孔洞的位置'''
    if len(hole_list) != 6:
        print(f"Warning: Hole Detection Failed. Detected Number:{len(hole_list)}")
    radius_list = []
    for seg in hole_list:
        cx, cy, r = locate_circle(seg.edge_point_i)
        if cx is not None and cy is not None and r is not None:
            hole_pos_list.append((int(cx), int(cy),r))
            radius_list.append(r)
    if len(hole_list) > 6:
        # 保留半径靠近均值的6个孔洞 # TODO：将偏离均值改为离群点检测
        mean = np.mean(radius_list)
        radius_err = [abs(r - mean) for r in radius_list]
        min_index = np.argsort(radius_err)[:6]  
        hole_pos_list = [hole_pos_list[i] for i in min_index]
    return gear_pos, hole_pos_list, gear_angle
    
def detect_img(img:np.ndarray, model, show=False) -> 'Tuple[Tuple[int,int,float],List[Tuple[int,int,float]],float]':
    '''
    处理单张图像，返回齿轮、孔洞位置和齿轮角度的检测结果
    > 参数：
    img: 输入图像，numpy数组格式
    show: 是否显示检测结果图像
    > 返回值:
    gear_pos: 齿轮位置和半径 (cx, cy, radius)
    hole_pos_list: 孔洞位置列表 [(cx, cy, radius), ...]
    gear_angle: 齿轮的旋转角度，范围 [0, 2*pi/z]
    '''
    # 裁剪使得两方向缩放比例相同
    img = img[:img.shape[0],:int(img.shape[0]*Const.IMG_SHAPE_O[1]/Const.IMG_SHAPE_O[0])]
    # print(f"Image shape: {img.shape}")
    # 使用YOLO模型进行预测
    results = model.predict(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    seg_list:List[SegmentResult] = []
    for i in range(len(boxes)):
        seg_list.append(SegmentResult(
            img = img,
            class_id = classes[i].item(),
            box = boxes[i],
            mask = masks[i]
        ))
    # print(f"Detected {len(seg_list)} segments.")

    # 绘制位置检测结果
    show_img = img.copy()

    gear_pos, hole_pos_list, gear_angle = locate_all_segment(seg_list, img=show_img, debug=True)
    # print(f"Gear Position: {gear_pos}, Holes: {hole_pos_list}, Gear Angle: {gear_angle:.2f} rad")
    if show:
        show_img = cv2.circle(show_img, gear_pos[:2], radius=5, color=(0, 0, 255), thickness=5)
        show_img = cv2.circle(show_img, gear_pos[:2], radius=int(gear_pos[2]), color=(0, 0, 255), thickness=5)
        for hole_pos in hole_pos_list:
            show_img = cv2.circle(show_img, hole_pos[:2], radius=5, color=(0, 0, 255), thickness=5)
            show_img = cv2.circle(show_img, hole_pos[:2], radius=int(hole_pos[2]), color=(0, 0, 255), thickness=5)
        # 绘制角度检测效果
        cv2.line (show_img, gear_pos[:2], (int(gear_pos[0] + 500 * np.cos(gear_angle)), int(gear_pos[1] + 500 * np.sin(gear_angle))), (0, 255, 255), 8)
        cv2.putText(show_img, f"Angle:{gear_angle*180/np.pi:.1f}deg", (gear_pos[0]+300, gear_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        show_img = cv2.resize(show_img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
        # # 绘制边缘提取结果
        # for seg in seg_list:
        #     if seg.class_name == "keyhole":
        #         continue
        #     x1, y1, x2, y2 = seg.xyxy_i
        #     print(f"{seg.class_name} bbox: ({x1}, {y1}), ({x2}, {y2})")
        #     show_img = seg.draw_edge(img=show_img, thickness=1, local=False, raw=False)
        return gear_pos, hole_pos_list, gear_angle, show_img

    return gear_pos, hole_pos_list, gear_angle

def main():

    imgdir_path = os.path.join(current_path, "imgset")
    for file in os.listdir(imgdir_path):
        if not file.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(imgdir_path,file))    
        gear_pos, hole_pos_list, gear_angle, img = detect_img(img, model, show=True)
        cv2.imshow(f"Detection Result - {file}", img)
        cv2.waitKey(0)
        break

if __name__ == "__main__":
    main()

