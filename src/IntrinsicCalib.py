import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from AuboControlLowLevel import AuboController
from MVSControl import MVSController
from ComputePose import ImageProcessor
from ConstConfig import Const
# TODO: 待更新，需要适配新的视觉算法
# ====== 配置参数 ======
ROBOT_IP = Const.Robot.IP
ROBOT_PORT = Const.Robot.PORT
INIT_POS = deepcopy(Const.Robot.CALIB_POS)        # 初始位置 [x, y, z] 单位: m
INIT_ORI = deepcopy(Const.Robot.CALIB_ORI)        # 初始姿态 [roll, pitch, yaw] 单位: rad
PLANE_AXIS = [0, 1]                         # 平面内移动的轴（如x和y）
STEP = 0.025                                # 步长（米）
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

YOLO_WEIGHTS = os.path.join(Const.Yolo.MODEL_DIE, 
                            Const.Yolo.YOLO_HOLE_WEIGHTS)
def generate_plane_positions(init_pos, axis, step, grid_size):
    """生成平面内9点标定的机械臂末端位置"""
    positions = []
    center = np.array(init_pos)
    for i in range(grid_size):
        for j in range(grid_size):
            pos = center.copy()
            pos[axis[0]] += (i - grid_size // 2) * step
            pos[axis[1]] += (j - grid_size // 2) * step
            positions.append(pos.tolist())
    return positions

def main():
    # 初始化机器人和相机
    aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT, enable_log=False)
    img_processor = ImageProcessor(model_weights=YOLO_WEIGHTS, show=True)
    mvs = MVSController()

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