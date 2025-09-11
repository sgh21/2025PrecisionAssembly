'''
使用标定板对相机内参进行标定，尚未完成
'''
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import os, sys
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("workspace:", workspace)
sys.path.append(workspace)

from AuboControlLowLevel import AuboController
from MVSControl import MVSController
from ComputePose import ImageProcessor
from ConstConfig import Const

# ====== 配置参数 ======
ROBOT_IP = Const.Robot.IP
ROBOT_PORT = Const.Robot.PORT
INIT_POS = deepcopy(Const.Robot.CALIB_POS)        # 初始位置 [x, y, z] 单位: m
INIT_ORI = deepcopy(Const.Robot.CALIB_ORI)        # 初始姿态 [roll, pitch, yaw] 单位: rad
PLANE_AXIS = [0, 1]                         # 平面内移动的轴（如x和y）
STEP = 0.015                                # 步长（米）
GRID_SIZE = 5                               # 3x3网格
DATASET_DIR = Const.Data.DATASET_DIR
SAVE_DIR = os.path.join(DATASET_DIR, 'intrinsic_calib')
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, 'intrinsic_calib.csv')
TIME_SLEEP = Const.Task.TIME_SLEEP  # 等待机械臂稳定的时间
JOINT_MAX_ACC = Const.Robot.JOINT_MAX_ACC
JOINT_MAX_VELC = Const.Robot.JOINT_MAX_VELC
END_MAX_ACC = Const.Robot.END_MAX_ACC
END_MAX_VELC = Const.Robot.END_MAX_VELC

YOLO_WEIGHTS = os.path.join(Const.Yolo.MODEL_DIR, 
                            Const.Yolo.YOLO_HOLE_WEIGHTS)

CALIB = Const.ClassInfo.CALIB_CLASS

import os
import sys
import glob
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(current_dir)
# sys.path.append(workspace)

IMAGE_DIR = os.path.join(workspace, "documents", "calib_board")


class Calibrator(object):
    def __init__(self, img_dir, shape_inner_corner, size_grid, visualization=True):
        """
        --parameters--
        img_dir: the directory that save images for calibration, str
        shape_inner_corner: the shape of inner corner, Array of int, (h, w)
        size_grid: the real size of a grid in calibrator, float
        visualization: whether visualization, bool
        """
        self.img_dir = img_dir
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.mat_intri = None # intrinsic matrix
        self.coff_dis = None # cofficients of distortion
        self.v_rot = None # rotation vectors
        self.v_trans = None # translation vectors
        # create the conner in world space
        w, h = shape_inner_corner
        # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
        # like (0,0,0), (1,0,0), (2,0,0) ...., (10,7,0)
        cp_int = np.zeros((w * h, 3), np.float32)
        cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        # print(cp_int)
        # cp_world: corner point in world space, save the coordinate of corner points in world space
        self.cp_world = cp_int * size_grid
        

        # images
        self.img_paths = []
        for extension in ["jpg", "png", "jpeg"]:
            self.img_paths += glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
        assert len(self.img_paths), "No images for calibration found!"
        
        # 按照文件名中的索引对 img_paths 进行排序
        # 不排序会后悔的，真的！！！
        def extract_index(file_path):
            file_name = os.path.basename(file_path)
            index = int(''.join(filter(str.isdigit, os.path.splitext(file_name)[0])))
            return index

        self.img_paths = sorted(self.img_paths, key=extract_index)
        print("Images for calibration have been loaded.", self.img_paths)

    def calibrate_camera(self):
        w, h = self.shape_inner_corner
        # criteria: only for subpix calibration, which is not used here
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        points_world = [] # the points in world space
        points_pixel = [] # the points in pixel space (relevant to points_world)
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the corners, cp_img: corner points in pixel space
            ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
            print(f"Processing image: {img_path}, corners found: {ret}")
            # if ret is True, save
            if ret:
                # optimize the corner points to subpix level
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cp_img = cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)

                points_world.append(self.cp_world)
                points_pixel.append(cp_img)
                # view the corners
                if self.visualization:
                    # 设置窗口名称
                    window_name = 'FoundCorners'
                    
                    # 创建一个可调整大小的窗口
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    
                    # 设置窗口大小，例如宽度为800，高度为600
                    cv2.resizeWindow(window_name, 1536, 1024)
                    cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
                    save_path = os.path.join(self.img_dir, "corners", os.path.basename(img_path))
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, img)
                    cv2.imshow(window_name, img)
                    cv2.waitKey(500)


        # calibrate the camera
        ret, mat_intri, coff_dis, v_rot, v_trans = cv2.calibrateCamera(points_world, points_pixel, gray_img.shape[::-1], None, None)

        total_error = 0
        for i in range(len(points_world)):
            points_pixel_repro, _ = cv2.projectPoints(points_world[i], v_rot[i],v_trans[i], mat_intri, coff_dis)
            error = cv2.norm(points_pixel[i], points_pixel_repro, cv2.NORM_L2)/len(points_pixel_repro)
            print("Error of reproject: {}".format(error))
            total_error += error
        print("Average error of reproject: {}".format(total_error/len(points_world)))

        self.mat_intri = mat_intri
        self.coff_dis = coff_dis
        self.v_rot = v_rot
        self.v_trans = v_trans

        # print(self.calculate_world_coordinates([self.mat_intri[0,2],self.mat_intri[1,2]],0))
        return mat_intri, coff_dis, v_rot, v_trans


# def main_client():
#     from RobotClientV2 import RobotClient
#     robot_client = RobotClient()
#     print('机器人客户端已初始化')

#     if robot_client.client_socket is None:
#         print("机器人客户端连接失败")
#         return

#     print("机器人客户端连接成功")
#     robot_client.set_robot_mode('stable')


#     robot_client.aubo.movel(INIT_POS, INIT_ORI, joint=True )
#     time.sleep(TIME_SLEEP)  # 等待机械臂稳定

#     records = []


#     robot_client.aubo.movel(INIT_POS, INIT_ORI, joint=True )
#     robot_client.disconnect()

def main():
    # # 初始化机器人和相机
    # aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT, enable_log=False)
    # img_processor = ImageProcessor(model_weights=YOLO_WEIGHTS, show=True)
    # mvs = MVSController()

    calibrator = Calibrator(
        img_dir=IMAGE_DIR,
        shape_inner_corner=(32,33), # 内角点个数
        size_grid=0.005,           # 标定板格子大小，单位米
        visualization=True
    )
    intrinsic, _, _, _ = calibrator.calibrate_camera()

    print("相机内参标定完成。\n内参：", intrinsic)


    # 机器人运动参数初始化
    aubo.set_joint_maxacc(JOINT_MAX_ACC)
    aubo.set_joint_maxvelc(JOINT_MAX_VELC)
    aubo.set_end_speed(END_MAX_VELC) 
    aubo.set_end_acc(END_MAX_ACC)
    print("设备初始化完成。")

    # 生成9点平面内标定位置
    positions = generate_plane_positions(INIT_POS, PLANE_AXIS, STEP, GRID_SIZE)
    ori = INIT_ORI

    aubo.movel(INIT_POS, INIT_ORI, joint=True )
    time.sleep(TIME_SLEEP)  # 等待机械臂稳定
    records = []
    for idx, pos in enumerate(positions):
        print(f"[{idx+1}/{GRID_SIZE * GRID_SIZE}] 移动到位置: {pos}")
        try:
            aubo.movel(pos, ori, joint=True)
            time.sleep(TIME_SLEEP)  # 等待机械臂稳定
            # 如果需要多次采集同一位置的图像
            img = mvs.get_image(debug=True)
            if img is not None:
                # 检测圆心像素坐标
                circle, _ = img_processor.detect_calib_hole(img, circle_fit_method='EdgeDrawing')
                if circle is not None:
                    u, v, r = circle
                    print(f"检测到圆心像素: ({u:.2f}, {v:.2f})")
                    # 在图像上画出圆心和圆
                    img_draw = img.copy()
                    cv2.circle(img_draw, (int(u), int(v)), int(r), (0,255,0), 2)
                    cv2.circle(img_draw, (int(u), int(v)), 3, (0,0,255), -1)
                else:
                    u, v, r = np.nan, np.nan, np.nan
                    print("未检测到圆心")
                # 保存图片
                img_name = f"calib_{idx+1:02d}.png"
                cv2.imwrite(os.path.join(SAVE_DIR, img_name), img_draw)
                # 记录数据
                records.append({
                    'idx': idx+1,
                    'x': pos[0] * 1000, 'y': pos[1] * 1000, 'z': pos[2] * 1000,
                    'u': u, 'v': v, 'r': r,
                    'img': img_name
                })
            else:
                print("采集图像失败，跳过。")
                records.append({
                    'idx': idx+1,
                    'x': pos[0] * 1000, 'y': pos[1] * 1000, 'z': pos[2] * 1000,
                    'u': np.nan, 'v': np.nan, 'r': np.nan,
                    'img': None
                })
        except Exception as e:
            print(f"采集第{idx+1}点时出错: {e}")
            records.append({
                'idx': idx+1,
                'x': pos[0] * 1000, 'y': pos[1] * 1000, 'z': pos[2] * 1000,
                'u': np.nan, 'v': np.nan, 'r': np.nan,
                'img': None
            })
            continue

    # 保存所有数据到CSV
    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"所有数据已保存到: {CSV_PATH}")
    # 拟合内参（以x-u为例，y-v同理）
        # 只用有效点
    valid = df.dropna(subset=['x', 'u', 'y', 'v'])
    X = valid[['x', 'y']].values  # shape (N,2)
    U = valid['u'].values         # shape (N,)
    V = valid['v'].values         # shape (N,)

    # 构造增广矩阵 [1, x, y]
    ones = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones, X])  # shape (N,3)

    # 拟合 u = u0 + a11*x + a12*y
    coeffs_u, _, _, _ = np.linalg.lstsq(X_aug, U, rcond=None)
    u0, a11, a12 = coeffs_u

    # 拟合 v = v0 + a21*x + a22*y
    coeffs_v, _, _, _ = np.linalg.lstsq(X_aug, V, rcond=None)
    v0, a21, a22 = coeffs_v

    # 预测
    U_pred = X_aug @ coeffs_u
    V_pred = X_aug @ coeffs_v
    err_u = np.abs(U_pred - U)
    err_v = np.abs(V_pred - V)
    mean_err_u = np.mean(err_u)
    mean_err_v = np.mean(err_v)

    # 记录误差到df
    valid = valid.copy()
    valid['u_pred'] = U_pred
    valid['v_pred'] = V_pred
    valid['err_u'] = err_u
    valid['err_v'] = err_v
    valid.to_csv(os.path.join(SAVE_DIR, 'intrinsic_calib_with_pred.csv'), index=False)

    print(f"\n最小二乘法拟合结果：")
    print(f"u = {u0:.4f} + {a11:.4f} * x + {a12:.4f} * y")
    print(f"v = {v0:.4f} + {a21:.4f} * x + {a22:.4f} * y")
    print(f"u轴平均绝对误差: {mean_err_u:.4f} 像素")
    print(f"v轴平均绝对误差: {mean_err_v:.4f} 像素")

    # 可视化
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.scatter(valid['x'], valid['u'], label='u (measured)', color='b')
    plt.scatter(valid['x'], valid['u_pred'], label='u (fitted)', color='r', marker='x')
    plt.xlabel('x (mm)')
    plt.ylabel('u (px)')
    plt.title('u-x fitted')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.scatter(valid['y'], valid['v'], label='v (measured)', color='b')
    plt.scatter(valid['y'], valid['v_pred'], label='v (fitted)', color='r', marker='x')
    plt.xlabel('y (mm)')
    plt.ylabel('v (px)')
    plt.title('v-y fitted')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'intrinsic_fit_affine.png'))
    plt.show()

if __name__ == "__main__":
    main()