from ultralytics import YOLO
from scipy.ndimage import binary_erosion
import torch
import numpy as np
import cv2
from typing import List, Tuple

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO("../yolo/runs/segment/train/weights/best.pt")  # Load a trained model

class Const:
    IMG_SHAPE_O = (448, 640, 3)  # YOLO输出图像尺寸
    GEAR_Z = 18                  # 齿轮的齿数
    PRESSURE_ANGLE = 20/180*np.pi# 齿轮压力角

class SegmentResult:
    '''存储一个分割对象的信息的类
    后缀为i的变量表示拍摄图像的属性，尺寸[2048,3072,3]
    后缀为o的变量表示模型输出的图像属性，尺寸[448,640,3]
    后缀为l的变量表示局部坐标系的属性
    '''
    class_dict = {
        0: "gear",      # 齿轮外轮廓
        1: "hole",      # 安装孔洞
        2: "keyhole"    # 齿轮内轮廓（齿轮键槽）
    }
    def __init__(self, img, class_id, box, mask: np.ndarray):
        self.position = None    # 位姿，待计算
        self.radius = None      # 半径，待计算
        self.angle = None       # 角度，待计算
        self.img_i = img
        self.img_o = cv2.resize(img, (640, 448))  # 输出图像尺寸
        self.class_id = class_id
        self.class_name = self.class_dict[class_id]
        self.box_i = box
        self.mask = mask
        self.scale_io = Const.IMG_SHAPE_O[0]/img.shape[0]  # scale_io<1，输出图像和输入图像的缩放比例
        self._cut_pic_with_box(padding = 30)
        self._calc_bound_with_mask()    # 方法1：yolo输出的mask边界
        self._calc_bound_with_box()     # 方法2: 用局部图像进行canny得到的边界
        self._filter_edge_point()       # 用mask边界过滤canny边界
    
    def _calc_bound_with_mask(self)->None:
        '''使用掩码计算边界点'''
        kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)
        eroded_mask = binary_erosion(self.mask, structure=kernel)    # 四边全1的掩码

        edge_mask = self.mask.astype(bool) & ~eroded_mask
        # print("Edge mask shape:", edge_mask.shape)
        self.mask_edge_point_o = np.column_stack(np.where(edge_mask))
    
    def _cut_pic_with_box(self, padding:int) -> None:
        '''根据边界框裁剪图片'''
        x1, y1, x2, y2 = self.box_i.astype(int)
        height, width= self.img_i.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width - 1, x2 + padding)
        y2 = min(height - 1, y2 + padding)
        self.xyxy_i = (x1, y1, x2, y2)
        self.xyxy_o = (int(x1 * self.scale_io), int(y1 * self.scale_io),
                       int(x2 * self.scale_io), int(y2 * self.scale_io))
        self.local_img_i = self.img_i[y1:y2, x1:x2]
    
    def _calc_bound_with_box(self) -> None:
        '''在局部图像上使用Canny边缘检测
        '''
        gray = cv2.cvtColor(self.local_img_i, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v = np.median(gray)
        if self.class_name == "gear":
            sigma = 0.1
        elif self.class_name == "hole":
            sigma = 0.33
        elif self.class_name == "keyhole":
            sigma = 0.5
        low_threshold = int(max(0, (1.0 - sigma) * v))
        high_threshold = int(min(255, (1.0 + sigma) * v))
    
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edge_point = np.column_stack(np.where(edges > 0))
        self.edge_point_l = edge_point  # 局部坐标系
        self.edge_point_i = edge_point + np.array([self.xyxy_i[1], self.xyxy_i[0]])  # 转换为原图坐标系

    def _filter_edge_point(self, min_distance=5)-> None:
        '''由mask边界过滤canny边界，排除canny边界中离mask边界距离大于阈值的点
        '''
        # print(self.mask_edge_point_o, self.edge_point_i)
        valid_points = []
        for p_i in self.edge_point_i:
            # p_i = p + np.array([self.xyxy_i[1], self.xyxy_i[0]])
            if np.any(np.linalg.norm(self.mask_edge_point_o / self.scale_io - p_i, axis=1) < min_distance):
                valid_points.append(p_i)
        self.edge_point_i = np.array(valid_points)

    def draw_edge(self,img=None, thickness=2, local=True) -> np.ndarray:
        '''
        在原图上绘制边界点
        输出：局部图尺寸/网络输出图尺寸 绘制边界后的图像
        '''
        if img is None:
            img = self.img_o.copy()
        if local:   # 输出为局部图像尺寸
            img = self.local_img_i.copy()
            for point in self.edge_point_l:
                cv2.circle(img, tuple(point[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img
        else:       # 输出为网络输出图尺寸
            for p in self.mask_edge_point_o:    # 绘制mask边界
                cv2.circle(img, tuple(p[::-1]), radius=1, color=(255, 0, 0), thickness=thickness)
            for p in self.edge_point_i:         # 绘制canny边界
                # p_o = (p + np.array([self.xyxy_i[1], self.xyxy_i[0]]))* self.scale_io
                p_o = p * self.scale_io
                p_o = p_o.astype(int)
                cv2.circle(img, tuple(p_o[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img

def filter_keyhole_cicle(seg:SegmentResult) -> 'List[List[int,int]]':
    '''对于keyhole类(齿轮内轮廓)，额外将bbox纵向2/3以下部分的点删去
    TODO：不是所有图像都沿纵向保留2/3，考虑引入角度参数
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

def calculate_gear_angle(gear_pos:Tuple[int,int,float], seg:SegmentResult) -> float:
    '''计算齿轮的旋转角度   # TODO:实现齿轮角度计算
    返回：范围 [0, 2*pi/z]
    '''
    if seg.class_name != "gear":
        return None
    edge_points = seg.edge_point_i
    cx, cy = gear_pos[:2]  # 齿轮中心位置
    edge_points_relative = edge_points - np.array([cy, cx])  # 转换为相对坐标系
    # 对边缘点进行下采样，增大采样点间距，计算齿顶角度均值时能够利用多个齿的齿顶
    edge_points_relative = fps_downsample_kmeans(edge_points_relative, num_samples=len(edge_points_relative)//10)
    u, v = edge_points_relative[:, 1], edge_points_relative[:, 0]
    rho, theta = np.sqrt(u**2+v**2), np.arctan2(v, u)   # 极坐标转换
    theta = np.mod(theta, 2 * np.pi)                    # 确保角度在 [0, 2*pi] 范围内
    print(f"theta range: {np.min(theta)} to {np.max(theta)}")
    print(f"rho range: {np.min(rho)} to {np.max(rho)}")
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
    # calculate_angle_with_involutes(theta, rho, edge_points_relative)

    # 方法二：暴力搜索rho的top-k点的theta，归一化到单齿角度范围后取均值
    k = 20  # 取前k个rho极值点
    k = min(size, k)
    top_k_indices = np.argsort(rho)[-k:]  # 获取rho的top-k点索引
    top_k_theta = theta[top_k_indices]  # 获取对应的theta值
    regularized_theta = np.mod(top_k_theta, 2 * np.pi/Const.GEAR_Z)  # 确保角度在 [0, 2*pi/z] 范围内
    print(f"Regularized theta range: {np.min(regularized_theta)} to {np.max(regularized_theta)}")
    print(regularized_theta)
    gear_angle_mean = np.mean(regularized_theta)  # 计算平均角度
    gear_angle_median = np.median(regularized_theta)  # 计算中位数角度
    print(f"Mean: {np.mean(regularized_theta)}, Median: {np.median(regularized_theta)}")

    img = seg.img_i.copy()  # 检查齿轮角度计算
    for p in edge_points_relative:
        x = int(cx + p[1])
        y = int(cy + p[0])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img = cv2.circle(img, (x, y), radius=1, color=(255, 0, 0), thickness=5)
    img = cv2.line (img, (cx, cy), (int(cx + 500 * np.cos(gear_angle_mean)), int(cy + 500 * np.sin(gear_angle_mean))), (0, 255, 255), 10)
    img = cv2.line (img, (cx, cy), (int(cx + 500 * np.cos(gear_angle_median)), int(cy + 500 * np.sin(gear_angle_median))), (0, 0, 255), 10)
    img = cv2.resize(img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
    cv2.imshow("Gear Angle", img)

    return gear_angle_mean

from sklearn.cluster import KMeans
import numpy as np

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
            
    # 1. 计算齿轮的位置
    if len(keyhole_list) != 1:
        print(f"WARNING: Keyhole Detection Failed. Detected Number:{len(keyhole_list)}")
    for seg in keyhole_list:
        filtered_points = filter_keyhole_cicle(seg)
        cx, cy, r = locate_circle(filtered_points)

        # if debug and img is not None: # 绘制keyhole边界
        #     for p in filtered_points:
        #         img = cv2.circle(img, tuple(p[::-1]), radius=1, color=(0, 255, 0), thickness=2)
        #     print(f"filtered keyhole points: {len(filtered_points)}")
        #     img = cv2.resize(img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
        #     cv2.imshow("Keyhole Circle", img)
        #     cv2.waitKey(0)
        if cx is not None and cy is not None and r is not None:
            gear_pos= (int(cx), int(cy),r)
    # # 使用霍夫圆变换更精确地定位齿轮位置 -> 效果不佳
    # for seg in keyhole_list:
    #     limg = seg.local_img_i.copy()
    #     gray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (9, 9), 2)
    #     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
    #                                param1=100, param2=50, minRadius=50, maxRadius=300)
    #     if circles is not None:
    #         for circle in circles[0, :]:
    #             cx, cy, r = circle
    #             ci = np.array([seg.xyxy_i[1], seg.xyxy_i[0]])+np.array([cx, cy])  # 转换为原图坐标系
    #             if np.linalg.norm(ci - np.array([gear_pos[1],gear_pos[0]])) > 10:  # 限制在齿轮位置附近
    #                 continue
    #             if debug:
    #                 limg = cv2.circle(limg, (int(cx), int(cy)), radius=int(r), color=(0, 255, 0), thickness=2)
    #                 print(f"Detected circle: center=({cx}, {cy}), radius={r}") 
    #         if limg is not None and debug:
    #             cv2.imshow("Hough Circle Detection", limg)
    #             cv2.waitKey(0)
    
    # 2. 计算齿轮的角度
    if len(gear_list) != 1:
        print(f"WARNING: Gear Detection Failed. Detected Number:{len(gear_list)}")
    for seg in gear_list:
        filter_gear_circle(gear_pos, seg)  # 过滤掉齿轮内轮廓点
        gear_angle = calculate_gear_angle(gear_pos, seg)
        pass # TODO: 计算齿轮的角度
    # 3. 计算孔洞的位置
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
    
def main():
    img = cv2.imread("img1.jpg")    
    # 裁剪使得两方向缩放比例相同
    img = img[:img.shape[0],:int(img.shape[0]*Const.IMG_SHAPE_O[1]/Const.IMG_SHAPE_O[0])]
    print(f"Image shape: {img.shape}")
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
    print(f"Detected {len(seg_list)} segments.")

    show_img = img.copy()
    # 绘制位置检测结果
    gear_pos, hole_pos_list, gear_angle = locate_all_segment(seg_list, img=show_img, debug=False)
    show_img = cv2.circle(show_img, gear_pos[:2], radius=5, color=(0, 0, 255), thickness=5)
    show_img = cv2.circle(show_img, gear_pos[:2], radius=int(gear_pos[2]), color=(0, 0, 255), thickness=5)
    for hole_pos in hole_pos_list:
        show_img = cv2.circle(show_img, hole_pos[:2], radius=5, color=(0, 0, 255), thickness=5)
        show_img = cv2.circle(show_img, hole_pos[:2], radius=int(hole_pos[2]), color=(0, 0, 255), thickness=5)
    # 绘制角度检测效果
    show_img = cv2.line (show_img, gear_pos[:2], (int(gear_pos[0] + 500 * np.cos(gear_angle)), int(gear_pos[1] + 500 * np.sin(gear_angle))), (0, 255, 255), 8)
    show_img = cv2.resize(show_img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
    # 绘制边缘提取结果
    for seg in seg_list:
        if seg.class_name == "keyhole":
            continue
        x1, y1, x2, y2 = seg.xyxy_i
        print(f"{seg.class_name} bbox: ({x1}, {y1}), ({x2}, {y2})")
        show_img = seg.draw_edge(img=show_img, thickness=1, local=False)
    cv2.imshow("Segmented Image", show_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

