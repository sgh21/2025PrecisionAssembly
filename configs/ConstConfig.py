import numpy as np
import cv2
from scipy.ndimage import binary_erosion

PI = np.pi

class Const:

    class Robot:
        """机器人相关配置"""
        # IP = '192.168.1.100'
        IP = '192.168.70.10'
        PORT = 8899
        # INIT_POS = [0.48657, 0.01178, 0.236420]
        # INIT_ORI = [PI, 0, PI/2]  # 初始姿态 [roll, pitch, yaw] 单位: rad
        # CALIB_POS = [0.48885, -0.079199, 0.236420]  # 校准位置 [x, y, z] 单位: m
        # CALIB_ORI = [PI, 0, PI/2]  # 校准姿态 [roll, pitch, yaw] 单位: rad

        INIT_POS = [-0.477483, -0.068571, 0.236420]  # 初始位置 [x, y, z] 单位: m
        INIT_ORI = [PI, 0, -PI/2]  # 初始姿态 [roll, pitch, yaw] 单位: rad
        CALIB_POS = [-0.48, -0.1576, 0.236420]
        CALIB_ORI = [PI, 0, -PI/2]  # 校准姿态 [roll, pitch, yaw] 单位: rad
        
        # HAND_IN_EYE_OFFSET = [-0.114547, -0.00155595, -0.20] # 手眼标定位置 [x, y, z] 单位: m
        JOINT_MAX_ACC = [0.3] * 6
        JOINT_MAX_VELC = [0.3] * 6
        END_MAX_ACC = 0.3
        END_MAX_VELC = 0.3
        JOINT_STABLE_ACC = [0.10] * 6
        JOINT_STABLE_VELC = [0.10] * 6
        END_STABLE_ACC = 0.10
        END_STABLE_VELC = 0.10
        JOINT_INSERT_ACC = [0.02] * 6  # 插入时的关节最大加速度
        JOINT_INSERT_VELC = [0.02] * 6
        END_INSERT_ACC = 0.02  # 插入时的末端最大加速度
        END_INSERT_VELC = 0.02
        POSE_ERROR_THRESHOLD = 5 * 1e-5  # 位置误差阈值（米） 0.05mm
        # CONTROLLER_INIT_ANGLE = 0.8728294 / 180 * PI  # 控制器初始角度，单位：deg
        USE_SEPARATE_HAND_IN_EYE_OFFSET = True # 每个孔使用各自的手眼标定结果
        # 比赛
        # X_OFFSET = -0.11  # 相机X轴偏移量（米） -2.0147132317361707
        # HAND_IN_EYE_OFFSET = [0.10967375, 0.00199195, -0.155]
        # 实验室
        X_OFFSET = 0.11  # 相机X轴偏移量（米） -2.0147132317361707
        HAND_IN_EYE_OFFSET = [-0.11344396, 0.00027828, -0.150]
        HAND_IN_EYE_OFFSET_LIST = [
            [-0.11341202, 0.00044193, -0.150],
            [-0.11335625, 0.0007936, -0.150],
            [-0.11348748, 0.00062911, -0.150],
            [-0.11342842, -0.0002757, -0.150],
            [-0.11342842, -0.0002757, -0.150],
            [-0.11342842, -0.0002757, -0.150]
        ]

        SEPARATE_CONTROLLER_INIT_ANGLE = False  # 是否分别使用不同的控制器初始角度
        CONTROLLER_INIT_ANGLE = -0.15 / 180 * PI
        CONTROLLER_INIT_ANGLE_LIST=[-5.71/180*PI, -1.21/180*PI, 3.29/180*PI, 0, 0, 0]
        STEP = 0.005
        DZ = 0.04
        # # 考虑板和相机的倾斜
        # INCLINE = False     # 是否考虑安装板和相机倾斜
        # T_camera2flange = None  # TODO:尚未完成标定
        # T_BOARD2BASE = None  # TODO:尚未完成标定1

        
    class Task:
        """任务相关配置"""
        TARGET_HOLE_IDX = 5
        TARGET_HOLE_IDX_LIST = [0, 1, 2]  # 目标孔索引列表
        # TARGET_HOLE_IDX_LIST = [ 0, 1, 2, 3, 4, 5]
        # TARGET_HOLE_IDX_LIST = [2,2,2]
        TIME_SLEEP = 0.8  # 等待机械臂稳定的时间
        WAITKEY = 30  # OpenCV窗口等待时间

    class Camera:
        IMG_SHAPE_SHOW = (1024, 1536, 3)
        # 比赛
        # INTRINSIC_A = [[-0.2816, 14.0124],  # 0.4191 px 0.4256 px
        #                [13.9848,  0.3358]]
        # 实验室
        INTRINSIC_A = [[-0.1842, -14.3617],  # 0.4191 px 0.4256 px
                       [-14.3878,  0.2278]]

        INTRINSIC_U0 = [1536, 1024]
        

    class Vision:
        """视觉相关配置"""
        HOST = 'localhost'
        PORT = 2025
        
        CUT_PADDING = 30  # 裁剪图片时的padding
        GEAR_SIGMA = 0.1  # 齿轮检测高斯滤波sigma
        HOLE_SIGMA = 0.33
        KEYHOLE_SIGMA = 0.5  # 孔检测高斯滤波sigma

        KEYHOLE_RANSAC_MAX_ITER = 1000  # RANSAC迭代次数
        KEYHOLE_RANSAC_THRESHOLD = 20  # RANSAC阈值
        KEYHOLE_RANSAC_MIN_INLIERS = 0.3  # RANSAC最小内点数

        HOLE_RANSAC_MAX_ITER = 1000  # RANSAC迭代次数
        HOLE_RANSAC_THRESHOLD = 100  
        HOLE_RANSAC_MIN_INLIERS = 0.5  # RANSAC最小内点数

        HOLE_RADIUS = 120  # 孔半径，单位：px

        N_SLICE_ITERS = 2  # 切片迭代次数

        # USE_RADIUS_SPLIT_CALIB_HOLE = True  # 使用半径区分标定圆和hole
        USE_RADIUS_SPLIT_CALIB_HOLE = False 
        RADIUS_THRESHOLD = HOLE_RADIUS-1  # 半径阈值，单位：px
        USE_CALIB_LIST = False  # 保留九个标定圆进行位姿计算(不稳定)

    class Gear:
        Z = 18
        PEAK_RADIUS = 60  # 齿顶圆半径，单位：mm
        KEYHOLE_MARK_RADIUS = 11    # SAM标记点所在圆的半径，单位：mm
        PRESSURE_ANGLE = 20/180*PI
        ERROR_POS = (0,0)
        ERROR_ANGLE = -1

    class Yolo:
        MODEL_DIR = r'./models'
        # YOLO_HOLE_WEIGHTS = 'yolov11s-seg-0819.pt'
        # YOLO_HOLE_WEIGHTS = 'yolov11s-seg-0910.pt'
        YOLO_HOLE_WEIGHTS = 'yolov11s-seg-0914-200.pt'
        YOLO_CONF = 0.8  # YOLO检测置信度阈值
    
    class Sam:
        MODEL_DIR = r'./models'
        SAM_MODEL_TYPE = 'vit_h'  # vit_b vit_h
        SAM_WEIGHTS = 'sam_vit_h_4b8939.pth'
        # SAM_WEIGHTS = 'sam_vit_b_01ec64.pth'
        SAM_CONF = 0.5  # SAM检测置信度阈值

    class Data:
        DATASET_DIR = r'./documents'

    class ClassInfo:
        """分类信息"""
        GEAR_CLASS = 'gear'
        HOLE_CLASS = 'hole'
        KEYHOLE_CLASS = 'keyhole'
        CALIB_CLASS = 'calib_circle'

        CLASS_DICT = {
            KEYHOLE_CLASS: 0,
            HOLE_CLASS: 1,
            GEAR_CLASS: 2,
            CALIB_CLASS: 3,
        }
        
        CLASS_NAME_DICT = {
            0: KEYHOLE_CLASS,
            1: HOLE_CLASS,
            2: GEAR_CLASS,
            3: CALIB_CLASS,
        }
        COLOR_LIST = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0]
        ]

class SegmentResult:
    '''存储一个分割对象的信息的类
    后缀为i的变量表示拍摄图像的属性，尺寸[2048,3072,3]
    后缀为l的变量表示局部坐标系的属性
    '''
    class_dict = Const.ClassInfo.CLASS_DICT
    def __init__(self, img, class_id, box, mask: np.ndarray):
        self.position = None    # 位姿，待计算
        self.radius = None      # 半径，待计算
        self.angle = None       # 角度，待计算
        self.img_i = img
        if type(class_id) == int:
            self.class_id = class_id
            self.class_name = list(self.class_dict.keys())[class_id]
        elif type(class_id) == str:
            self.class_name = class_id
            # self.class_id = self.class_dict[class_id]
        self.box_i = box
        self.mask = mask
        self._cut_pic_with_box(padding = Const.Vision.CUT_PADDING)  # 根据边界框裁剪图片
        self._calc_bound_with_mask()    # 方法1：yolo输出的mask边界
        self._calc_bound_with_box()     # 方法2: 用局部图像进行canny得到的边界
        self._filter_edge_point(min_distance=5)       # 用mask边界过滤canny边界
    
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
        self.local_img_i = self.img_i[y1:y2, x1:x2]
    
    def _calc_bound_with_box(self) -> None:
        '''在局部图像上使用Canny边缘检测
        '''
        gray = cv2.cvtColor(self.local_img_i, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v = np.median(gray)
        if self.class_name == Const.ClassInfo.GEAR_CLASS:
            sigma = Const.Vision.GEAR_SIGMA
        elif self.class_name == Const.ClassInfo.HOLE_CLASS:
            sigma = Const.Vision.HOLE_SIGMA
        elif self.class_name == Const.ClassInfo.KEYHOLE_CLASS:
            sigma = Const.Vision.KEYHOLE_SIGMA
        else:
            sigma = 0.33
        low_threshold = int(max(0, (1.0 - sigma) * v))
        high_threshold = int(min(255, (1.0 + sigma) * v))
    
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edge_point = np.column_stack(np.where(edges > 0))
        self.edge_point_l = edge_point  # 局部坐标系
        self.edge_point_i = edge_point + np.array([self.xyxy_i[1], self.xyxy_i[0]])  # 转换为原图坐标系
        self.edge_point_i_raw = self.edge_point_i.copy()  # 保存原始边界点位置
        # # 将边缘绘制在原图可视化
        # img_vis = self.img_i.copy()
        # for p in self.edge_point_i:
        #     cv2.circle(img_vis, tuple(p[::-1]), radius=1, color=(0, 255, 255), thickness=1)
        # self.img_edge_vis = img_vis  # 保存可视化结果
        # # 可视化窗口缩放显示
        # win_name = f"Edge Detection - {self.class_name}"
        # scale = 0.5  # 缩放比例，可根据需要调整
        # img_show = cv2.resize(img_vis, (0, 0), fx=scale, fy=scale)
        # cv2.imshow(win_name, img_show)
        # cv2.waitKey(0)  # 等待按键
        # cv2.destroyWindow(win_name)

    def _filter_edge_point(self, min_distance=5)-> None:
        '''由mask边界过滤canny边界，排除canny边界中离mask边界距离大于阈值的点
        '''
        # print(self.mask_edge_point_o, self.edge_point_i)
        valid_points = []
        for p_i in self.edge_point_i:
            # p_i = p + np.array([self.xyxy_i[1], self.xyxy_i[0]])
            if np.any(np.linalg.norm(self.mask_edge_point_o  - p_i, axis=1) < min_distance):
                valid_points.append(p_i)
        self.edge_point_i = np.array(valid_points)

    def draw_edge(self,img=None, thickness=2, local=False, raw=False) -> np.ndarray:
        '''
        在原图上绘制边界点
        输出：局部图尺寸/网络输出图尺寸 绘制边界后的图像
        '''
        if img is None:
            img = self.img_i.copy()
        if local:   # 输出为局部图像尺寸
            img = self.local_img_i.copy()
            for point in self.edge_point_i:
                cv2.circle(img, tuple(point[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img
        else:       # 输出为网络输出图尺寸
            for p in self.mask_edge_point_o:    # 绘制mask边界
                cv2.circle(img, tuple(p[::-1]), radius=1, color=(255, 0, 0), thickness=thickness)
            if raw:
                list = self.edge_point_i_raw
            else:
                list = self.edge_point_i
            for p in list:         # 绘制canny边界
                cv2.circle(img, tuple(p[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img