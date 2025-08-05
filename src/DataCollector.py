import os
import numpy as np
import cv2
import time
import copy
from configs.config import *
from ConstConfig import Const
from Transform import *
from AuboControlLowLevel import AuboController
from MVSControl import MVSController


# ====== 配置参数 ======
ROBOT_IP = Const.Robot.IP
ROBOT_PORT = Const.Robot.PORT
INIT_POS = copy.deepcopy(Const.Robot.INIT_POS)          # 初始位置 [x, y, z] 单位: m
INIT_ORI = copy.deepcopy(Const.Robot.INIT_ORI)          # 初始姿态 [roll, pitch, yaw] 单位: rad
POS_RANGE = [0.05, 0.05, 0.0]                     # 每个方向最大扰动范围 ±m
ORI_RANGE = [0, 0, 10]                            # 姿态扰动范围 ±deg
NUM_SAMPLES = 100                                  # 采集数量
DATASET_DIR = r'./documents/dataset/0803/images'  # 保存路径

os.makedirs(DATASET_DIR, exist_ok=True)

def random_pose_around(init_pos, init_ori, pos_range, ori_range):
    # 位置扰动
    pos_delta = (np.random.rand(3) - 0.5) * 2 * np.array(pos_range)
    # 姿态扰动（角度转弧度）
    ori_delta = (np.random.rand(3) - 0.5) * 2 * np.deg2rad(ori_range)
    pos = np.array(init_pos) + pos_delta
    ori = np.array(init_ori) + ori_delta
    return pos.tolist(), ori.tolist()

def main():
    # 初始化机器人和相机
    aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT)
    mvs = MVSController()
    print("设备初始化完成。")

    os.makedirs(DATASET_DIR, exist_ok=True)

    # 机器人运动参数初始化
    aubo.set_joint_maxacc([0.2] * 6)
    aubo.set_joint_maxvelc([0.2] * 6)
    aubo.set_end_speed(0.2) 
    aubo.set_end_acc(0.2)

    use_current_waypoint = input("是否使用当前位姿作为初始位置？(y/n): ").strip().lower()
    if use_current_waypoint == 'y':
        waypoint = aubo.get_current_waypoint()
        INIT_POS = waypoint['pos']
        INIT_ORI = quaternion_standard2rpy(waypoint['ori'])
        print(f"使用当前位姿作为初始位姿: pos={INIT_POS}, ori={INIT_ORI}")
    else:
        INIT_POS = copy.deepcopy(Const.Robot.INIT_POS)          # 初始位置 [x, y, z] 单位: m
        INIT_ORI = copy.deepcopy(Const.Robot.INIT_ORI)          # 初始姿态 [roll, pitch, yaw] 单位: rad
        print(f"使用预设初始位姿: pos={INIT_POS}, ori={INIT_ORI}")
    aubo.movel(INIT_POS, INIT_ORI, joint=True)  # 移动到初始位置

    for idx in range(NUM_SAMPLES):
        idx += 100
        # 生成随机位姿
        pos, ori = random_pose_around(INIT_POS, INIT_ORI, POS_RANGE, ORI_RANGE)
        print(f"[{idx+1}/{NUM_SAMPLES}] 移动到位置: {pos}, 姿态: {ori}")
        try:
            aubo.movel(pos, ori, joint=True)
            time.sleep(0.8)  # 等待机械臂稳定
            img = mvs.get_image()
            if img is not None:
                img_name = f"image_{idx+1:04d}.png"
                img_path = os.path.join(DATASET_DIR, img_name)
                cv2.imwrite(img_path, img)
                print(f"已保存图像: {img_path}")
            else:
                print("采集图像失败，跳过。")
        except Exception as e:
            print(f"采集第{idx+1}张图像时出错: {e}")
            continue
    aubo.movel(INIT_POS, INIT_ORI, joint=True)  # 移动到初始位置
    aubo.disconnect()
    mvs.close_device()
    print("数据采集完成。")

if __name__ == "__main__":
    main()