'''
综合使用YOLO和SAM实现的ComputePose
TODO:筛选hole的策略更新为到齿轮中心最近的六个并过滤距离的离群值
'''
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

import numpy as np
from typing import List, Tuple
from copy import deepcopy
import cv2
from configs.ConstConfig import Const, SegmentResult

# * : begin define utility functions

def fit_circle(points):
    """
    Fit a circle to 2D points using linear least squares (Kåsa method).
    Returns (xc, yc, r).
    """
    pts = np.asarray(points)
    x = pts[:,0]
    y = pts[:,1]
    # Solve x^2 + y^2 + A x + B y + C = 0
    D = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    A, B, C = np.linalg.lstsq(D, b, rcond=None)[0]
    xc = -A/2
    yc = -B/2
    r = np.sqrt(np.maximum(xc**2 + yc**2 - C, 0))
    return xc, yc, r


def ransac_circle(points, max_trials=1000, threshold=2.0, min_inliers=0.5):
    if len(points) < 3:
        return None, None, None
    points = np.asarray(points)
    N = len(points)
    best_inliers = []

    for _ in range(max_trials):
        idx = np.random.choice(N, 3, replace=False)
        sample = points[idx]
        try:
            xc, yc, r = fit_circle(sample)
        except Exception:
            continue
        dists = np.abs(np.sqrt((points[:,0]-xc)**2 + (points[:,1]-yc)**2) - r)
        inliers = dists < threshold
        if np.sum(inliers) > max(min_inliers*N, np.sum(best_inliers)):
            best_inliers = inliers

    if np.sum(best_inliers) >= 3:
        xc, yc, r = fit_circle(points[best_inliers])
        return xc, yc, r
    else:
        return None, None, None

def remove_outliers_mad(data, threshold=2.0):
    """
    使用中位数绝对偏差(MAD)方法去除离群值
    """
    if len(data) < 3:
        return None
    
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return None
    
    modified_z_scores = 0.6745 * (data - median) / mad
    mask = np.abs(modified_z_scores) < threshold
    return mask


def HoughCircleMethod(gray_img, return_all = False):
    # 使用HoughCircles方法
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=100, param2=30, minRadius=50, maxRadius=200)
    print(f"Use HoughCircles detected circles: {circles}")
    if circles is not None and len(circles[0]) > 0:
        if return_all:
            return [(float(x), float(y), float(r)) for x, y, r in circles[0]]
        x, y, r = circles[0][0]
        return (float(x), float(y), float(r))
    else:
        return None
    
def EdgeDrawingMethod(gray_img, return_all = False):
    # 边缘检测（EdgeDrawing）
    try:
        ed = cv2.ximgproc.createEdgeDrawing()
        edParams = cv2.ximgproc_EdgeDrawing_Params()
        # edParams.EllipseFitErrThreshold = 1
        ed.setParams(edParams)
        ed.detectEdges(gray_img)
        ed.detectLines()
        ellipses = ed.detectEllipses()
        print(f"Use EdgeDrawing detected ellipses: {ellipses}")
        if ellipses is not None and len(ellipses) > 0:
            for e in ellipses[0]:
                if e[2]==0:
                    e[2] = (e[3]+e[4])/2
            if return_all:
                return [(float(e[0]), float(e[1]), float(e[2])) for e in ellipses[0]]
            x, y, r = ellipses[0][0][:3]
            return (float(x), float(y), float(r))
        else:
            # EdgeDrawing可用但检测失败
            return HoughCircleMethod(gray_img, return_all=return_all)
    except AttributeError:
        # EdgeDrawing不可用时，退化为HoughCircles
        return HoughCircleMethod(gray_img, return_all=return_all)

# * : end define utility functions



class LocateHole:
    
    def fliter_boxes_by_class(self, 
                              seg_list:List[SegmentResult], 
                              target_cls= 'hole'):
        """
        筛选YOLO检测结果中的指定类别的边界框。

        参数:
            seg_list: yolo检测结果的SegmentResult对象
            target_cls: 目标类别

        返回:
            筛选后的边界框列表
        """

        target_list = []
        for seg in seg_list:
            if seg.class_name == target_cls:
                target_list.append(seg)
        
        if len(target_list) == 0:
            print(f"\033[33mWARNING: No {target_cls} detected.\033[0m")
            return []
        
        return target_list

    def fliter_boxes_by_expected_num(self, 
                                     seg_list:List[SegmentResult], 
                                     expected_num=6
                                     ):
        """
        根据六边形分布，去除离群孔
        参数:
            seg_list: yolo检测结果的SegmentResult对象
            expected_num: 目标类别个数
        返回:
            排序后的(center_x, center_y)列表和对应的编号列表
        """
        # 兼容传入boxes对象或xyxy数组
        hole_seg_list = self.fliter_boxes_by_class(seg_list, target_cls = Const.ClassInfo.HOLE_CLASS)
        
        centers = []
        for seg in hole_seg_list:
            # 计算中心点
            x1, y1, x2, y2 = seg.xyxy_i
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))

        # 对中心点进行外接RANSAC圆拟合
        centers_np = np.array(centers)
        cx, cy, r = ransac_circle(centers_np, 
                                  max_trials=Const.Vision.HOLE_RANSAC_MAX_ITER, 
                                  threshold=Const.Vision.HOLE_RANSAC_THRESHOLD, 
                                  min_inliers=Const.Vision.HOLE_RANSAC_MIN_INLIERS)

        # TODO: 如果检测结果超过六个，使用六边形拟合筛选
        # 策略：以所有中心点的外接RANSAC圆为中心，计算到中心的距离，取距离最接近均值距离的expected_num个点
        if len(centers_np) > expected_num:
            dists = np.sqrt((centers_np[:, 0] - cx) ** 2 + (centers_np[:, 1] - cy) ** 2)
            mean_dist = np.mean(dists)
            # 取距离中心距离最接近均值的expected_num个点
            selected_idx = np.argsort(np.abs(dists - mean_dist))[:expected_num]
            selected_centers = centers_np[selected_idx]
            hole_seg_list = [hole_seg_list[i] for i in selected_idx]

        return hole_seg_list

    def complete_hexagon(self, points, indices, tol_ang=np.pi/7, tol_rad=0.4):
        """
        Recover all 6 vertices of an (approximately) regular hexagon.
        
        points: list of (x,y) existing vertices (no particular order)
        indices: list of original indices corresponding to each point
        tol_ang: max angular deviation (rad) to match an existing point to a slot
        tol_rad: max radial deviation (fraction of radius) to accept an existing point
        """
        pts = np.asarray(points)
        
        if len(pts) < 3:
            raise ValueError("Need at least 3 points to estimate hexagon")
        
        # 1) Fit circle
        xc, yc, r = fit_circle(pts)
        
        # 生成理论六变形点坐标
        angles = np.arctan2(pts[:,1] - yc, pts[:,0] - xc) % (2*np.pi)
        ref_angle = np.min(angles)
        slot_angles = (ref_angle + np.arange(6) * (np.pi/3)) % (2*np.pi)
        assigned = [None]*6
        assigned_idx = [None]*6
        # 2) Assign points to slots
        for i, (x, y) in enumerate(pts):
            ang = angles[i]
            rad = np.sqrt((x-xc)**2 + (y-yc)**2)
            rad_err = np.abs(rad - r)/r
            rad_err = 0
            # 计算周期性角度差
            diffs = np.abs((ang - slot_angles + np.pi) % (2*np.pi) - np.pi)
            slot = np.argmin(diffs)
            ang_err = diffs[slot]
            if ang_err < tol_ang and rad_err < tol_rad and assigned[slot] is None:
                assigned[slot] = (x, y)
                assigned_idx[slot] = indices[i]
        # 缺失点的索引用-1补全
        # next_idx = max(indices)+1 if indices else 0
        for k in range(6):
            if assigned[k] is None:
                xg = xc + r * np.cos(slot_angles[k])
                yg = yc + r * np.sin(slot_angles[k])
                assigned[k] = (xg, yg)
                assigned_idx[k] = -1
               
        assert len(assigned) == 6, "Hexagon should have exactly 6 vertices"
        return assigned, assigned_idx

    def sort_hexagon_centers(self, centers, indices):
        """
        对六边形分布的圆孔进行排序，最上方为0，逆时针编号。
        参数:
            centers: (center_x, center_y)列表
            indices: 对应的编号列表
        返回:
            排序后的(center_x, center_y)列表和对应的编号列表
        """

        centers_np = np.array(centers)
        cx, cy = np.mean(centers_np, axis=0)

        # 计算极角，y轴向下，取负号
        angles = np.arctan2(-(centers_np[:,1] - cy), centers_np[:,0] - cx)
        # 最上方（y最小）为0号，逆时针排序
        top_idx = np.argmin(centers_np[:,1])
        ref_angle = angles[top_idx]
        norm_angles = (angles - ref_angle) % (2 * np.pi)
        sort_order = np.argsort(norm_angles)
        sorted_centers = [centers[idx] for idx in sort_order]
        sorted_indices = [indices[idx] for idx in sort_order]
        return sorted_centers, sorted_indices
    
    def filter_hole_by_radius(self, 
                              xyxy_i: Tuple[int, int, int, int],
                              circles: List[Tuple[float, float, float]],
                              radius )-> List[Tuple[float, float, float]]:
                        # 使用半径在80像素做初步筛选
        heuristic_circles = [c for c in circles if abs(c[2]-radius)<10]
        if len(heuristic_circles) > 0:
            circles = heuristic_circles
        else:
            print(f"\033[33mWARNING: No circle found with radius close to {radius}.\033[0m")
            circle = (xyxy_i[0] + circles[0][0], xyxy_i[1] + circles[0][1], circles[0][2]) 
            return circle
        # 根据边界框位置和长宽筛选最优圆
        error =[]
        bx1, by1, bx2, by2 = xyxy_i
        bw, bh = bx2 - bx1, by2 - by1
        for c in circles:
            x, y, r = c
            error.append(
                abs(x-bw/2)
                + abs(y-bh/2)
                + abs(r - radius)
            )
        circle = circles[np.argmin(error)] if len(circles) > 0 else None
        # print(f"Min Error:{min(error)}")
        circle = (xyxy_i[0] + circle[0], xyxy_i[1] + circle[1], circle[2]) if circle is not None else (None, None, None)
        return circle
    
    def locate_hole(self,
                seg_list:List[SegmentResult], 
                show_img: np.ndarray = None,
                circle_fit_method:str = 'EdgeDrawing'):
        hole_seg_list = self.fliter_boxes_by_expected_num(seg_list, expected_num=6)
        if len(hole_seg_list) == 0:
            print("\033[33mWARNING: No hole detected.\033[0m")
            return None
        
        # 构造seg_list的中心点列表和对应的索引
        centers = []
        indices = []
        for i, seg in enumerate(hole_seg_list):
            # 计算中心点
            x1, y1, x2, y2 = seg.xyxy_i
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
            indices.append(i)
        
        if len(centers) < 6 :
            centers, indices = self.complete_hexagon(centers, indices)

        # 对六边形分布的圆孔进行排序
        sorted_centers, sorted_indices = self.sort_hexagon_centers(centers, indices)

        # 根据sorted_indices重新排序hole_seg_list
        sorted_hole_seg_list = []
        for i in sorted_indices:
            if i < 0:
                sorted_hole_seg_list.append(None)  # -1表示缺失点
            else:
                sorted_hole_seg_list.append(hole_seg_list[i])
        
        hole_list = []
        for i, seg in enumerate(sorted_hole_seg_list):
            if seg is None:
                hole_list.append((sorted_centers[i][0], sorted_centers[i][1], None))  # 缺失点
            else:
                limg = seg.local_img_i
                gray_img = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY) if len(limg.shape) == 3 else limg
                # 使用指定的圆拟合方法
                circles = None
                if circle_fit_method == 'EdgeDrawing':
                    circles = EdgeDrawingMethod(gray_img, return_all=True)
                if circles is None or len(circles) == 0:
                    # 使用其它方法或者检测失败时，使用HoughCircles
                    circles = HoughCircleMethod(gray_img, return_all=True)
                # TODO: 测试并检查代码逻辑
                if circles is None or len(circles) == 0:
                    # 如果圆拟合失败，则可能是圆存在残缺
                    print("\033[33mWARNING: Circle fitting failed, using bounding box center and radius.\033[0m")
                    circle = ((seg.xyxy_i[0] + seg.xyxy_i[2]) / 2, (seg.xyxy_i[1] + seg.xyxy_i[3]) / 2, Const.Vision.HOLE_RADIUS)
                    hole_list.append(circle)
                else:
                    circle = self.filter_hole_by_radius(seg.xyxy_i, circles, radius=Const.Vision.HOLE_RADIUS)
                    hole_list.append(circle)
        # 显示结果 
        if show_img is not None:
            for i, (cx, cy, r) in enumerate(hole_list):
                r = r if r is not None else 80
                cv2.circle(show_img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                cv2.putText(show_img, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        return hole_list, show_img
    

    def locate_calib_circle_list(self,
                            seg_list:List[SegmentResult],
                            show_img: np.ndarray = None,
                            circle_fit_method:str = 'EdgeDrawing'):
        calib_seg_list = self.fliter_boxes_by_class(seg_list, target_cls=Const.ClassInfo.CALIB_CLASS)
        pos_list = [((seg.xyxy_i[0] + seg.xyxy_i[2]) / 2, (seg.xyxy_i[1] + seg.xyxy_i[3]) / 2, (seg.xyxy_i[2]-seg.xyxy_i[0]+seg.xyxy_i[3]-seg.xyxy_i[1]) / 4) for seg in calib_seg_list]
        x, y = [p[0] for p in pos_list], [p[1] for p in pos_list]
        middle_pos = [np.median(x), np.median(y)]
        distance = np.mean([np.max(x)-np.min(x), np.max(y)-np.min(y)]) / 2
        ideal_pos = np.array([
            [middle_pos[0] - distance, middle_pos[1] - distance],
            [middle_pos[0], middle_pos[1] - distance],
            [middle_pos[0] + distance, middle_pos[1] - distance],
            [middle_pos[0] - distance, middle_pos[1]],
            [middle_pos[0], middle_pos[1]],
            [middle_pos[0] + distance, middle_pos[1]],
            [middle_pos[0] - distance, middle_pos[1] + distance],
            [middle_pos[0], middle_pos[1] + distance],
            [middle_pos[0] + distance, middle_pos[1] + distance],
        ], dtype=int)
        result = pos_list.copy()

        # result = np.ones((9,3)).tolist()        
        # for calib_seg in calib_seg_list:
        #     limg = calib_seg.local_img_i
            
        #     gray_img = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY) if len(limg.shape) == 3 else limg
        #     # 使用指定的圆拟合方法
        #     circles = None
        #     if circle_fit_method == 'EdgeDrawing':
        #         circles = EdgeDrawingMethod(gray_img, return_all=True)
        #     if circles is None or len(circles) == 0:
        #         # 使用其它方法或者检测失败时，使用HoughCircles
        #         circles = HoughCircleMethod(gray_img, return_all=True)
        #     # TODO: 测试并检查代码逻辑
        #     if circles is None or len(circles) == 0:
        #         print("\033[33mWARNING: Circle fitting failed, using bounding box center and radius.\033[0m")
        #         x1, y1, x2, y2 = calib_seg.xyxy_i
        #         # 如果圆拟合失败，使用边界框中心和半径
        #         cx = (x1 + x2) / 2
        #         cy = (y1 + y2) / 2
        #         r = max(x2 - x1, y2 - y1) / 2
        #         circle = (cx, cy, r)
        #     else:
        #         circle = self.filter_hole_by_radius(calib_seg.xyxy_i, circles, radius=Const.Vision.HOLE_RADIUS)
        #     nearest_idx = np.argmin([(circle[0]-ip[0])**2+(circle[1]-ip[1])**2 for ip in ideal_pos])
        #     result[nearest_idx] = circle

        if show_img is not None:
            for circle in result:
                if circle[0] is not None and circle[1] is not None:
                    cv2.circle(show_img, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 0, 255), 2)
                    cv2.putText(show_img, "Calib Hole", (int(circle[0]), int(circle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
        return result



    def locate_calib_circle(self, 
                            seg_list:List[SegmentResult], 
                            show_img: np.ndarray = None,
                            circle_fit_method:str = 'EdgeDrawing'):
        
        calib_seg_list = self.fliter_boxes_by_class(seg_list, target_cls=Const.ClassInfo.CALIB_CLASS)

        if len(calib_seg_list) != 1:
            print(f"\033[33mWARNING: Found {len(calib_seg_list)} calibration holes.\033[0m")
        
        calib_seg = calib_seg_list[0] if len(calib_seg_list) > 0 else None

        if calib_seg is None:
            return None
            # raise ValueError("No calibration hole detected.")
        
        limg = calib_seg.local_img_i
        
        gray_img = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY) if len(limg.shape) == 3 else limg
        # 使用指定的圆拟合方法
        circles = None
        if circle_fit_method == 'EdgeDrawing':
            circles = EdgeDrawingMethod(gray_img, return_all=True)
        if circles is None or len(circles) == 0:
            # 使用其它方法或者检测失败时，使用HoughCircles
            circles = HoughCircleMethod(gray_img, return_all=True)
        # TODO: 测试并检查代码逻辑
        if circles is None or len(circles) == 0:
            print("\033[33mWARNING: Circle fitting failed, using bounding box center and radius.\033[0m")
            x1, y1, x2, y2 = calib_seg.xyxy_i
            # 如果圆拟合失败，使用边界框中心和半径
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            r = max(x2 - x1, y2 - y1) / 2
            circle = (cx, cy, r)
        else:
            circle = self.filter_hole_by_radius(calib_seg.xyxy_i, circles, radius=Const.Vision.HOLE_RADIUS)

        if show_img is not None:
            if circle[0] is not None and circle[1] is not None:
                cv2.circle(show_img, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 0, 255), 2)
                cv2.putText(show_img, "Calib Hole", (int(circle[0]), int(circle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        return circle
    
class LocateGear:

    def filter_gear_circle(self, 
                           gear_pos:'Tuple[int,int,float]', 
                           seg:SegmentResult) -> SegmentResult:
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
        # 去除mask边界点中的内轮廓点
        center = np.array([gear_pos[1], gear_pos[0]])   # 转换为网络输出图坐标系
        radius_inner = gear_pos[2] 
        edge_points_relative = seg.mask_edge_point_o - center  # 转换为相对坐标系
        radius = np.linalg.norm(edge_points_relative, axis=1)
        mask = radius > radius_inner * 2  # 保留半径大于内轮廓半径n倍的点
        seg.mask_edge_point_o = seg.mask_edge_point_o[mask]
        return seg
    
    def calculate_gear_angle_waist(self,
                                    gear_pos:Tuple[int,int,float], 
                                    seg:SegmentResult,
                                    show_img: np.ndarray = None,
                                    mask_edge: bool = True) -> float:
        '''计算齿轮的旋转角度，使用两侧渐开线上截取点的方法
        返回：范围 [0, 2*pi/z]
        '''
        def gear_angle_normalize(angle, tooth_range):
            return angle % tooth_range
        if seg.class_name != "gear":
            return None
        if mask_edge:
            edge_points = seg.mask_edge_point_o
        else:
            edge_points = seg.edge_point_i
        cx, cy = gear_pos[:2]  # 齿轮中心位置
        edge_points_relative = edge_points - np.array([cy, cx])  # 转换为相对坐标系
        # 对边缘点进行下采样，增大采样点间距，计算齿顶角度均值时能够利用多个齿的齿顶
        # edge_points_relative = fps_downsample_kmeans(edge_points_relative, num_samples=len(edge_points_relative)//10)
        u, v = edge_points_relative[:, 1], edge_points_relative[:, 0]
        rho, theta = np.sqrt(u**2+v**2), np.arctan2(v, u)   # 极坐标转换
        theta = np.mod(theta, 2 * np.pi)                    # 确保角度在 [0, 2*pi] 范围内
        # print(f"Shape rho: {rho.shape}, theta: {theta.shape}, edge_points_relative: {edge_points_relative.shape}")
        # print(f"theta range: {np.min(theta)} to {np.max(theta)}")
        # print(f"rho range: {np.min(rho)} to {np.max(rho)}")
        tooth_height = (np.max(rho) - np.min(rho))          # 齿高
        radius_pitch = (np.max(rho) + np.min(rho)) / 2      # 分度圆半径
        # radius_base = radius_pitch * np.cos(Const.Gear.PRESSURE_ANGLE)    # 基圆半径

        # 方法：将每个齿切成两半，2z段分别寻找最接近radius_mark的点
        # 找到最大rho所在theta
        tooth_range = 2 * np.pi / Const.Gear.Z  # 单齿角度范围
        theta_valley = np.mod(theta[np.argmin(rho)], tooth_range)  # 最小rho对应的theta
        # theta_valley = np.mod(theta[np.argmax(rho)]+1/2*tooth_range, tooth_range)  # 最小rho对应的theta

        for i in range(Const.Vision.N_SLICE_ITERS):  # 为解决切片初值敏感问题，迭代两次
            if i == 0:
                theta_ini = theta_valley
            else:
                theta_ini = theta_res - tooth_range/2
            # 根据齿轮齿数，将theta切分成z个部分
            theta_left, theta_right = [], []
            rho_left, rho_right = [], []

            for i in range(Const.Gear.Z*2):
                valley_right = (i%2==0)     # 顺时针，True表示当前分段是从齿底到齿顶
                start_angle = (theta_ini + i * tooth_range/2) % (2 * np.pi)
                end_angle = (theta_ini + (i + 1) * tooth_range/2) % (2 * np.pi)
                if start_angle < end_angle:
                    mask = (theta >= start_angle) & (theta < end_angle)
                else:# 处理跨越0度的情况
                    mask = (theta >= start_angle) | (theta < end_angle)
                # 提取该区间内的点
                if valley_right:
                    theta_right.append(theta[mask])
                    rho_right.append(rho[mask])
                else:
                    theta_left.append(theta[mask])
                    rho_left.append(rho[mask])

            # print(f"齿根左侧点数：{[len(r) for r in rho_left]}, 右侧点数：{[len(r) for r in rho_right]}")

            radius_mark = radius_pitch+tooth_height/6 # 选取标记点的圆半径

            theta_mark_left = []  # 左半齿的标记点角度
            theta_mark_right = []

            if show_img is not None :
                for i, r in enumerate(rho_left):
                    error = np.abs(r - radius_mark)
                    if len(error) == 0:
                        continue
                    # if min(error) > 20:
                    #     continue
                    idx_left = np.argmin(error)  # 找到最接近radius_mark的点
                    theta_mark = theta_left[i][idx_left]
                    theta_mark_left.append(theta_mark)  # 获取对应的theta值
                    cv2.circle(show_img, (int(cx + r[idx_left] * np.cos(theta_mark)), int(cy + r[idx_left] * np.sin(theta_mark))), 2, (255, 0, 255), 10)
                for i, r in enumerate(rho_right):
                    error = np.abs(r - radius_mark)
                    if len(error) == 0:
                        continue
                    # if min(error) > 20:
                    #     continue
                    idx_right = np.argmin(error)
                    theta_mark = theta_right[i][idx_right]
                    theta_mark_right.append(theta_mark)  # 获取对应的theta值
                    cv2.circle(show_img, (int(cx + r[idx_right] * np.cos(theta_mark)), int(cy + r[idx_right] * np.sin(theta_mark))), 2, (255, 0, 255), 10)
            
            # 归一化到单齿范围
            offset = theta_ini
            regularized_theta_left = np.mod(np.array(theta_mark_left) - offset, tooth_range)  # 确保角度在 [0, 2*pi/z] 范围内
            regularized_theta_right = np.mod(np.array(theta_mark_right) - offset, tooth_range)

            mask_left = remove_outliers_mad(regularized_theta_left, threshold=2.0)  # 使用MAD方法去除离群值
            mask_right = remove_outliers_mad(regularized_theta_right, threshold=2.0)
            if mask_left is None or mask_right is None:
                return Const.Gear.ERROR_ANGLE
            regularized_theta_left = regularized_theta_left[mask_left]  # 过滤离群值
            regularized_theta_right = regularized_theta_right[mask_right]

            theta_mean_left = np.mean(regularized_theta_left)  # 计算左半齿的平均角度
            theta_mean_right = np.mean(regularized_theta_right)
            if theta_mean_left < theta_mean_right:
                theta_mean_left += tooth_range  # 确保左半齿的平均角度大于右半齿
            theta_res = np.mod((theta_mean_left + theta_mean_right) / 2 + offset, tooth_range) # 计算整体平均角度
        
        # 检查检测结果是齿顶还是齿根，如果检测成齿根，则+齿轮半齿角度
        theta_peak = [theta_res+i*tooth_range for i in range(0, Const.Gear.Z)] # 当前输出（认为是齿顶）的角度对应的18个齿顶角度
        if theta_res > tooth_range/2:
            theta_valley = [theta_res-tooth_range/2+i*tooth_range for i in range(0, Const.Gear.Z)]
        else:
            theta_valley = [theta_res+tooth_range/2+i*tooth_range for i in range(0, Const.Gear.Z)]
        peak_radius_sum, valley_radius_sum = 0, 0
        for t in theta_peak:
            idx = np.argmin(np.abs(theta - t))
            peak_radius_sum += rho[idx]
        for t in theta_valley:
            idx = np.argmin(np.abs(theta - t))
            valley_radius_sum += rho[idx]
        if valley_radius_sum > peak_radius_sum:   # 如果检测出的齿顶半径比齿根半径小，则切换
            theta_res = gear_angle_normalize(theta_res + tooth_range/2, tooth_range)

        return theta_res
    
    
    
    def locate_gear(self, 
                    seg_list:List[SegmentResult], 
                    show_img: np.ndarray = None, 
                    circle_fit_method:str = 'EdgeDrawing') -> Tuple[Tuple[int, int, float], float]:
        """
        处理齿轮检测结果，返回齿轮位置和角度 px, rad
        """

        gear_pos = Const.Gear.ERROR_POS+(0,) # 齿轮位置，原图坐标系，默认值(0,0)
        gear_angle = Const.Gear.ERROR_ANGLE   # 范围 [0, 2*pi/z]，默认值-1
        gear_list = [seg for seg in seg_list if seg.class_name == Const.ClassInfo.GEAR_CLASS]
        keyhole_list = [seg for seg in seg_list if seg.class_name == Const.ClassInfo.KEYHOLE_CLASS]
        print(f"Detected {len(gear_list)} gears, {len(keyhole_list)} keyholes.")
        
        '''1. 计算齿轮的位置'''
        if len(keyhole_list) != 1:
            print("\033[33mWARNING: Keyhole Detection Failed. Detected Number:{}\033[0m".format(len(keyhole_list)))
        
        # *: 取置信度最高的keyhole作为齿轮位置
        # TODOL: 需要考察keyhole_list的第一个是否为置信度最高的
        # 直接使用mask边缘点进行圆拟合
        keyhole_seg = keyhole_list[0]
        edge_points = keyhole_seg.edge_point_i    # 使用边缘点（不过滤），直接进行圆拟合
        # *: opencv 边界点的排列是H,W,C 因此返回结果应该修改为 cy, cx, r
        cy, cx, r = ransac_circle(edge_points, 
                                  max_trials=Const.Vision.KEYHOLE_RANSAC_MAX_ITER,
                                  threshold=Const.Vision.KEYHOLE_RANSAC_THRESHOLD, 
                                  min_inliers=Const.Vision.KEYHOLE_RANSAC_MIN_INLIERS)
        print(f"Keyhole circle fitting result: center=({cx}, {cy}), radius={r}")
        if cx is not None and cy is not None and r is not None:
            gear_pos= (int(cx), int(cy), r) # 全局坐标系下的齿轮位置
        else:
            print("\033[33mWARNING: Keyhole circle fitting failed.\033[0m")
            gear_pos = ((keyhole_seg.xyxy_i[0] + keyhole_seg.xyxy_i[2]) / 2, \
                        (keyhole_seg.xyxy_i[1] + keyhole_seg.xyxy_i[3]) / 2, 50 ) # 默认位置和半径
            
        # 使用霍夫圆变换或者EdgeDrawing对局部图像处理更精确地定位齿轮位置
        limg = deepcopy(keyhole_seg.local_img_i)
        gray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY) if len(limg.shape) == 3 else limg
        # !: 高斯模糊可能会影响定位精度
        gray = cv2.GaussianBlur(gray, (5, 5), 2)
        if circle_fit_method == 'EdgeDrawing':
            circles =  EdgeDrawingMethod(gray, return_all = True)
        else:
            circles =  HoughCircleMethod(gray, return_all = True)
        
        if circles is not None:
            error = []
            for circle in circles:
                cx, cy, r = circle
                ci = np.array([keyhole_seg.xyxy_i[0], keyhole_seg.xyxy_i[1]])+np.array([cx, cy])  # 转换为原图坐标系
                # # !:  0139遇到bug
                # if np.linalg.norm(ci - np.array(gear_pos[:2])) > 50:
                #     error.append(1e6)  # 距离过远，认为是错误检测
                #     print(f"\033[31mWARNING: Circle detection out of range: {ci} vs {gear_pos[:2]}\033[0m")
                # 计算误差：距离齿轮位置的距离 + 半径误差
                # 这里的10是一个权重系数，可以根据实际情况调整
                error.append(10*np.linalg.norm(ci - np.array(gear_pos[:2]))+abs(r+10 - gear_pos[2]))

            best_circle = circles[np.argmin(error)]
            gear_pos = (int(keyhole_seg.xyxy_i[0] + best_circle[0]), int(keyhole_seg.xyxy_i[1] + best_circle[1]), best_circle[2])
        
        '''2. 计算齿轮的角度'''
        if len(gear_list) != 1:
            print(f"\033[31mWARNING: Gear Detection Failed. Detected Number:{len(gear_list)}\033[0m")
        
        gear_seg = gear_list[0]
        self.filter_gear_circle(gear_pos, gear_seg)  # 过滤掉齿轮内轮廓点
        gear_angle = self.calculate_gear_angle_waist(gear_pos, gear_seg, show_img = show_img)
        if  show_img is not None:
            # 显示齿轮位置和角度
            cx, cy, r = int(gear_pos[0]), int(gear_pos[1]), int(gear_pos[2])
            cv2.circle(show_img, (int(cx), int(cy)), radius=int(r), color=(0, 0, 255), thickness=2)
            # 绘制直线显示角度
            angle = gear_angle  # 单位: 弧度
            x2 = int(cx + r * np.cos(angle) * 4)
            y2 = int(cy + r * np.sin(angle) * 4)
            cv2.line(show_img, (cx, cy), (x2, y2), (255, 0, 255), 3)
            # 显示分割掩码
            def visualize_mask(image, mask, color=(0,255,0), contour_only=False):
                """可视化SAM分割结果"""
                mask = mask.astype(bool)
                if contour_only:
                    # print(f"#####################{np.count_nonzero(mask)/mask.size}#####################")
                    # 创建彩色掩码覆盖层
                    colored_mask = np.zeros_like(image)
                    colored_mask[mask] = list(color)  # 绿色掩码
                    # 添加半透明掩码覆盖
                    image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                # 绘制掩码轮廓
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 5)
                return image

            show_img = visualize_mask(show_img, gear_seg.mask, contour_only=False)
            show_img = visualize_mask(show_img, keyhole_seg.mask, color=(255,0,0), contour_only=False)
            cv2.putText(show_img, f"Gear Angle: {gear_angle*180/np.pi:.1f} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            cv2.putText(show_img, f"Gear Pos: ({gear_pos[0]}, {gear_pos[1]})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)


        return gear_pos, gear_angle, show_img

class ImageProcessor:
    def __init__(self, device: str, yolo_model_weights: str, model_weights: str, model_type: str, show: bool = False, waitkey: int = 100):
        self.yolo_model = YOLO(yolo_model_weights)
        self.model = SamPredictor(sam_model_registry[model_type](checkpoint=model_weights).to(device))
        self.show = show
        self.waitkey = waitkey
        self.gear_locator = LocateGear()
        self.hole_locator = LocateHole()
    
    def yolo_predict(self, img: np.ndarray) -> List[SegmentResult]:
        results = self.yolo_model.predict(img, retina_masks = True, conf = Const.Yolo.YOLO_CONF)
        if len(results) == 0 or results[0].masks==None:
            print("No valid detection results found.")
            return Const.Gear.ERROR_POS+(0,), [], Const.Gear.ERROR_ANGLE, None
        
        if self.show:
            img_show = results[0].plot()  # 获取可视化结果
            cv2.namedWindow("YOLO Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO Result", Const.Camera.IMG_SHAPE_SHOW[1], Const.Camera.IMG_SHAPE_SHOW[0])  # 你想要的尺寸
            cv2.imshow("YOLO Result", img_show)
            cv2.waitKey(self.waitkey)  # 等待按键事件，0表示无限等待

        boxes = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        seg_list:List[SegmentResult] = []
        for i in range(len(boxes)):
            # 创建SegmentResult对象 
            # 每个对象会根据yolo检测结果进行裁剪 local_img_i
            # 并分别根据局部图像和原图像进行边缘检测 
            seg_list.append(SegmentResult(
                img = img,
                class_id = classes[i].item(),
                box = boxes[i],
                mask = masks[i]
            ))
        return seg_list
    
    def yolo_sam_predict(self, img: np.ndarray) -> List[SegmentResult]:
        '''
        用于检测齿轮位置，有线使用yolo，在检测失效情况下使用sam辅助检测
        '''
        mark_points = []
        results = self.yolo_model.predict(img, retina_masks = True, conf = Const.Yolo.YOLO_CONF)
        if len(results) == 0 or results[0].masks==None:
            print("No valid detection results found.")
            return Const.Gear.ERROR_POS+(0,), []
        
        if self.show:
            img_show = results[0].plot()  # 获取可视化结果
            cv2.namedWindow("YOLO Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO Result", Const.Camera.IMG_SHAPE_SHOW[1], Const.Camera.IMG_SHAPE_SHOW[0])  # 你想要的尺寸
            cv2.imshow("YOLO Result", img_show)
            cv2.waitKey(self.waitkey)  # 等待按键事件，0表示无限等待

        boxes = results[0].boxes.xyxy.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        seg_list:List[SegmentResult] = []
        for i in range(len(boxes)):
            # 创建SegmentResult对象 
            # 每个对象会根据yolo检测结果进行裁剪 local_img_i
            # 并分别根据局部图像和原图像进行边缘检测 
            seg_list.append(SegmentResult(
                img = img,
                class_id = classes[i].item(),
                box = boxes[i],
                mask = masks[i]
            ))
            
            # ### 测试SAM对keyhole分割效果，注释则在YOLO失效时才使用SAM
            # if seg_list[-1].class_name == Const.ClassInfo.KEYHOLE_CLASS:
            #     seg_list.pop()  # 删除yolo检测到的keyhole
            # ###
        has_gear = any([seg.class_name == Const.ClassInfo.GEAR_CLASS for seg in seg_list])
        has_keyhole = any([seg.class_name == Const.ClassInfo.KEYHOLE_CLASS for seg in seg_list])
        if has_gear and not has_keyhole:
            gear_seg = [seg for seg in seg_list if seg.class_name == Const.ClassInfo.GEAR_CLASS][0]
            x1, y1, x2, y2 = gear_seg.xyxy_i
            center = [(x1+x2)/2, (y1+y2)/2]  # 中心点
            radius_a = (x2-x1+y2-y1)/4 # 齿顶半径估计值
            radius_mark = radius_a / Const.Gear.PEAK_RADIUS * Const.Gear.KEYHOLE_MARK_RADIUS #SAM标记点的半径估计值
            for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
                px = int(center[0] + radius_mark * np.cos(angle))
                py = int(center[1] + radius_mark * np.sin(angle))
                mark_points.append((px, py))  # 1表示前景点
            point_coords = np.array(mark_points)
            point_labels = np.array([1]*len(mark_points))  # 1 for foreground
            self.model.set_image(img)
            masks, scores, logits = self.model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = masks[0]
            box = np.array([np.min(np.where(mask)[1]), np.min(np.where(mask)[0]), np.max(np.where(mask)[1]), np.max(np.where(mask)[0])])  # x1, y1, x2, y2
            seg_list.append(SegmentResult(
                img=img,
                class_id=Const.ClassInfo.KEYHOLE_CLASS,
                box=box,
                mask=mask
            ))
            

        return seg_list, mark_points
    
    def visualize_sam_result(self, image, box, mask):
        """可视化SAM分割结果"""
        result = image.copy()
        
        # 创建彩色掩码覆盖层
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [0, 255, 0]  # 绿色掩码
        
        # 添加半透明掩码覆盖
        result = np.where(mask == 1, result * 0.9 + colored_mask[mask] * 0.1, result)

        # 绘制掩码轮廓
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        return result

    def sam_predict(self, 
                    img: np.ndarray, 
                    mark_points:List[Tuple[int,int,int]],
                    class_ids:List[int]) -> np.ndarray:
        self.model.set_image(img)
        seg_list=[]
        for i in range(len(mark_points)):
            point_coords = np.array([[mark_points[i][0], mark_points[i][1]]])
            point_labels = np.array([mark_points[i][2]])  # 1 for foreground
            masks, scores, logits = self.model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = masks[0]
            box = np.array([np.min(np.where(mask)[1]), np.min(np.where(mask)[0]), np.max(np.where(mask)[1]), np.max(np.where(mask)[0])])  # x1, y1, x2, y2
            seg_list.append(SegmentResult(
                img=img,
                class_id=class_ids[i],
                box=box,
                mask=mask
            ))
            img_res = self.visualize_sam_result(img, box, mask)
            img_res = cv2.resize(img_res, (Const.Camera.IMG_SHAPE_SHOW[1]//2, Const.Camera.IMG_SHAPE_SHOW[0]//2))
            cv2.imshow("SAM Result", img_res)
        return seg_list
    
    @ staticmethod
    def _show_img(show_img, waitkey: int = 100):
        """显示当前处理的图像"""
        cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection Result", Const.Camera.IMG_SHAPE_SHOW[1], Const.Camera.IMG_SHAPE_SHOW[0])  # 你想要的尺寸
        cv2.imshow("Detection Result", show_img)
        cv2.waitKey(waitkey)  # 等待按键事件，0表示无限等待


    def process_image(self, img: np.ndarray, circle_fit_method: str = 'EdgeDrawing') -> Tuple[Tuple[int, int, float], List[Tuple[int, int, float]], float, Tuple[float, float, float]]:
        '''
        处理单张图像，返回齿轮、孔洞位置和齿轮角度的检测结果
        > 参数：
        img: 输入图像，numpy数组格式
        model_weights: 模型权重文件路径
        show: 是否显示检测结果图像
        > 返回值:
        gear_pos: 齿轮位置和半径 (cx, cy, radius)
        hole_pos_list: 孔洞位置列表 [(cx, cy, radius), ...]
        gear_angle: 齿轮的旋转角度，范围 [0, 2*pi/z]
        calib_hole: 标定孔洞位置和半径 (cx, cy, radius)
        '''
        show_img = deepcopy(img) 
        # 进行YOLO检测
        seg_list, mark_points = self.yolo_sam_predict(img)
        for mark in mark_points:
            cv2.circle(show_img, (mark[0], mark[1]), 5, (0, 255, 0), -1)

        # gear_num = sum([1 for seg in seg_list if seg.class_name == Const.ClassInfo.GEAR_CLASS])
        # keyhole_num = sum([1 for seg in seg_list if seg.class_name == Const.ClassInfo.KEYHOLE_CLASS])
        # hole_num = sum([1 for seg in seg_list if seg.class_name == Const.ClassInfo.HOLE_CLASS])
        # calib_num = sum([1 for seg in seg_list if seg.class_name == Const.ClassInfo.CALIB_CLASS])
        # print(f"Detected {gear_num} gears, {keyhole_num} keyholes, {hole_num} holes, {calib_num} calib holes.")
        # cv2.putText(show_img, f"Detected {gear_num} gears, {keyhole_num} keyholes, {hole_num} holes, {calib_num} calib holes.", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 5)

        assert len(seg_list) > 0, "No valid segment results found"

        # 处理齿轮检测
        try:
            gear_pos, gear_angle, show_img = self.gear_locator.locate_gear(
                seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method
            ) if self.gear_locator else (Const.Gear.ERROR_POS, Const.Gear.ERROR_ANGLE, show_img)
        except Exception as e:
            print(f"\033[31m[ERROR] Gear locating failed: {e}\033[0m")
            gear_pos, gear_angle = Const.Gear.ERROR_POS, Const.Gear.ERROR_ANGLE


        # 处理孔洞检测
        try:
            hole_list, show_img = self.hole_locator.locate_hole(
                seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method
            ) if self.hole_locator else ([], show_img)
        except Exception as e:
            print(f"\033[31m[ERROR] Hole locating failed: {e}\033[0m")
            hole_list = []

        # 处理标定孔洞检测
        try:
            if Const.Vision.USE_CALIB_LIST:
                calib_hole = self.hole_locator.locate_calib_circle_list(
                    seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method
                ) if self.hole_locator else (None, None, None)
            else:
                calib_hole = self.hole_locator.locate_calib_circle(
                    seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method
                ) if self.hole_locator else (None, None, None)
        except Exception as e:
            print(f"\033[31m[ERROR] Calib hole locating failed: {e}\033[0m")
            calib_hole = (None, None, None)

        if self.show and show_img is not None:
            self._show_img(show_img, waitkey=self.waitkey)

        return gear_pos, hole_list, gear_angle, calib_hole, show_img  # 返回齿轮位置和角度、孔洞列表、标定孔洞
    
    def detect_gear(self,
                    img: np.ndarray,
                    circle_fit_method: str = 'EdgeDrawing') -> Tuple[Tuple[int, int, float], float]:
        """ 检测齿轮位置和角度 """

        show_img = deepcopy(img) 
        seg_list, mark_points = self.yolo_sam_predict(img)

        for mark in mark_points:
            cv2.circle(show_img, (mark[0], mark[1]), 5, (0, 255, 0), -1)
        
        assert len(seg_list) > 0, "No valid segment results found"
        # 处理齿轮检测
        gear_pos, gear_angle, show_img = self.gear_locator.locate_gear(seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method) if self.gear_locator else (Const.Gear.ERROR_POS, Const.Gear.ERROR_ANGLE)
        if self.show and show_img is not None:
            self._show_img(show_img, waitkey=self.waitkey)

        return gear_pos, gear_angle, show_img  # 返回齿轮位置和角度 (cx, cy, radius), angle
    
    def detect_hole(self, 
                    img: np.ndarray, 
                    circle_fit_method: str = 'EdgeDrawing') -> List[Tuple[int, int, float]]:

        show_img = deepcopy(img) 
        seg_list = self.yolo_predict(img)

        
        assert len(seg_list) > 0, "No valid segment results found"
        # 处理孔洞检测
        hole_list, show_img = self.hole_locator.locate_hole(seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method) if self.hole_locator else []

        if self.show and show_img is not None:
            self._show_img(show_img, waitkey=self.waitkey)

        return hole_list, show_img
    
    def detect_calib_hole(self, 
                          img: np.ndarray, 
                          circle_fit_method: str = 'EdgeDrawing') -> Tuple[float, float, float]:

        show_img = deepcopy(img) 
        seg_list = self.yolo_predict(img)

        assert len(seg_list) > 0, "No valid segment results found"

        
        for seg in seg_list:
            print(f"Class: {seg.class_name}, Box: {seg.xyxy_i}, Box Height: {seg.xyxy_i[3]-seg.xyxy_i[1]}, Width: {seg.xyxy_i[2]-seg.xyxy_i[0]}")

        # 处理标定孔洞检测
        if Const.Vision.USE_CALIB_LIST:
            calib_hole = self.hole_locator.locate_calib_circle_list(seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method) if self.hole_locator else (None, None, None)
        else:
            calib_hole = self.hole_locator.locate_calib_circle(seg_list=seg_list, show_img=show_img, circle_fit_method=circle_fit_method) if self.hole_locator else (None, None, None)

        if self.show and show_img is not None:
            self._show_img(show_img, waitkey=self.waitkey)
        
        return calib_hole, show_img  # 返回圆心坐标和半径 (cx, cy, radius)

def test_circle_stability(num_samples=100, interval=0.2):
    """
    Continuously capture num_samples frames, detect the circle center, and plot the fluctuation.
    """
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from MVSControl import MVSController
    model_weights = Const.Yolo.YOLO_HOLE_WEIGHTS

    # 检查模型权重
    model_weights_path = os.path.join(Const.Yolo.MODEL_DIE, model_weights)
    image_processor = ImageProcessor(model_weights=model_weights_path, show=False)
    mvs_control = MVSController()
    xs, ys = [], []
    valid_idx = []
    print("Start continuous detection of circle center... Press Ctrl+C to interrupt.")
    try:
        for i in range(num_samples):
            img = mvs_control.get_image()
            if img is None:
                print(f"[{i+1}] No image captured")
                xs.append(None)
                ys.append(None)
                time.sleep(interval)
                continue

            gear_pos, gear_angle, show_img = image_processor.detect_gear(img, circle_fit_method='EdgeDrawing')
            result = gear_pos 
            if result:
                x, y, r = result
                print(f"[{i+1}] Center: ({x:.2f}, {y:.2f})")
                xs.append(x)
                ys.append(y)
                valid_idx.append(i)
            else:
                print(f"[{i+1}] No circle detected")
                xs.append(None)
                ys.append(None)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Detection interrupted.")
    finally:
        mvs_control.close_device()

    # Only count valid points
    xs_valid = [v for v in xs if v is not None]
    ys_valid = [v for v in ys if v is not None]
    valid_idx = [i for i, v in enumerate(xs) if v is not None and ys[i] is not None]
    if xs_valid and ys_valid:
        xs_valid = np.array(xs_valid)
        ys_valid = np.array(ys_valid)
        dx = xs_valid - np.mean(xs_valid)
        dy = ys_valid - np.mean(ys_valid)
        fluct = np.sqrt(dx**2 + dy**2)
        print(f"Std of center: x={np.std(xs_valid):.2f}, y={np.std(ys_valid):.2f}, Mean fluctuation={np.mean(fluct):.2f} px")
        # Plot
        plt.figure(figsize=(10,8))
        plt.subplot(3,1,1)
        plt.plot(valid_idx, xs_valid, label='x')
        plt.ylabel('x (px)')
        plt.title('Circle Center X Fluctuation')
        plt.grid(True)
        plt.subplot(3,1,2)
        plt.plot(valid_idx, ys_valid, label='y', color='orange')
        plt.ylabel('y (px)')
        plt.title('Circle Center Y Fluctuation')
        plt.grid(True)
        plt.subplot(3,1,3)
        plt.plot(valid_idx, fluct, label='Fluctuation', color='green')
        plt.xlabel('Frame Index')
        plt.ylabel('Fluctuation (px)')
        plt.title('Circle Center Overall Fluctuation')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid circle center detected.")

def vision_test():
    import os
    from MVSControl import MVSController
    model_weights = Const.Yolo.YOLO_HOLE_WEIGHTS

    # 检查模型权重
    model_weights_path = os.path.join(Const.Yolo.MODEL_DIE, model_weights)
    image_processor = ImageProcessor(model_weights=model_weights_path, show=True, waitkey = 20)
    mvs_control = MVSController()

    while True:
        img = mvs_control.get_image()
        if img is None:
            print("No image captured, retrying...")
            continue
        gear_pos, hole_list, gear_angle, calib_hole = image_processor.process_image(img, circle_fit_method='EdgeDrawing')
        print(f"Gear position: {gear_pos}")
        print(f"Hole list: {hole_list}")
        print(f"Gear angle: {gear_angle * 180 / np.pi:.2f} degrees")

def yolo_test():
    import os
    from MVSControl import MVSController
    model_weights = Const.Yolo.YOLO_HOLE_WEIGHTS
    modefl_dir = Const.Yolo.MODEL_DIE

    # 检查模型权重
    model_weights_path = os.path.join(modefl_dir, model_weights)
    if not os.path.exists(model_weights_path):
        print(f"\033[31mERROR: Model weights not found: {model_weights_path}\033[0m")
        return
    mvs_control = MVSController()
    img_processor = ImageProcessor(model_weights=model_weights_path, show=True, waitkey=20)
    while True:
        img = mvs_control.get_image()
        if img is None:
            print("No image captured, retrying...")
            continue
        result = img_processor.yolo_predict(img)
        if not result:
            print("No valid detection results found.")
            continue

def main():
    import os

    # 修改为你的图片文件夹和模型权重路径
    data_dir = Const.Data.DATASET_DIR
    test_dir = os.path.join(data_dir, 'yolo_0803/train/images')
    model_weights = Const.Yolo.YOLO_HOLE_WEIGHTS

    # 检查模型权重
    model_weights_path = os.path.join(Const.Yolo.MODEL_DIE, model_weights)
    if not os.path.exists(model_weights_path):
        print(f"\033[31mERROR: Model weights not found: {model_weights_path}\033[0m")
        return

    # 获取所有图片文件
    img_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not img_files:
        print(f"\033[31mERROR: No image files found in: {test_dir}\033[0m")
        return

    # 创建处理器
    processor = ImageProcessor(model_weights=model_weights_path, show=True, waitkey=10)
    img_show_list = [138, 139, 140]  # 用于存储处理后的图像
    # for img_name in img_files:
    for idx in img_show_list:
        img_name = img_files[idx]  # 仅处理指定的图片
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"\033[31mERROR: Failed to load image: {img_path}\033[0m")
            continue

        print(f"\nProcessing: {img_name}")
        gear_pos, hole_list, gear_angle, calib_hole = processor.process_image(img, circle_fit_method='EdgeDrawing')
        
        print(f"Gear position: {gear_pos}")
        print(f"Hole list: {hole_list}")
        print(f"Gear angle: {gear_angle * 180 / np.pi:.2f} degrees")

        key = cv2.waitKey(10)  # 等待按键事件，0表示无限等待
        if key == 27:  # ESC键
            print("Exiting...")
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # main()
    # test_circle_stability(num_samples=100, interval=0.2)
    vision_test()
    # yolo_test()