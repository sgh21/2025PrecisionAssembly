import cv2
import time

import os, sys
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workspace)
from utils.AuboControlLowLevel import AuboController
from utils.Transform import *
from utils.MVSControl import MVSController
from configs.ConstConfig import Const


ROBOT_IP = Const.Robot.IP
ROBOT_PORT = Const.Robot.PORT

INIT_POS = Const.Robot.INIT_POS
INIT_ORI = Const.Robot.INIT_ORI

aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT, enable_log=True)
mvs = MVSController()

img_dir = './documents/captures/'
os.makedirs(img_dir, exist_ok=True)
# 清空目录下的文件
for f in os.listdir(img_dir):
    os.remove(os.path.join(img_dir, f))

def set_robot_mode(mode: str = 'fast'): 
    """设置机械臂运动模式"""
    if mode == 'fast':
        aubo.set_joint_maxacc(Const.Robot.JOINT_MAX_ACC)
        aubo.set_joint_maxvelc(Const.Robot.JOINT_MAX_VELC)
        aubo.set_end_speed(Const.Robot.END_MAX_VELC)
        aubo.set_end_acc(Const.Robot.END_MAX_ACC)
    elif mode == 'stable':
        aubo.set_joint_maxacc(Const.Robot.JOINT_STABLE_ACC)
        aubo.set_joint_maxvelc(Const.Robot.JOINT_STABLE_VELC)
        aubo.set_end_speed(Const.Robot.END_STABLE_VELC)
        aubo.set_end_acc(Const.Robot.END_STABLE_ACC)
    elif mode == 'insert':
        aubo.set_joint_maxacc(Const.Robot.JOINT_INSERT_ACC)
        aubo.set_joint_maxvelc(Const.Robot.JOINT_INSERT_VELC)
        aubo.set_end_speed(Const.Robot.END_INSERT_VELC)
        aubo.set_end_acc(Const.Robot.END_INSERT_ACC)
    elif mode == 'spin':
        aubo.set_joint_maxacc(Const.Robot.JOINT_SPIN_ACC)
        aubo.set_joint_maxvelc(Const.Robot.JOINT_SPIN_VELC)
        aubo.set_end_speed(Const.Robot.END_SPIN_VELC)
        aubo.set_end_acc(Const.Robot.END_SPIN_ACC)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def main():
    # aubo.connect()
    set_robot_mode('fast')
    print("机械臂已连接，当前为快速模式。")
    aubo.movel(INIT_POS, INIT_ORI, joint=False)
    print("机械臂已移动到初始位置。")

    time.sleep(1)

    dest_pos = [INIT_POS[0]+0.05, INIT_POS[1]+0.05, INIT_POS[2]]
    dest_ori = INIT_ORI

    aubo.movel(dest_pos, dest_ori, joint=False)
    print("机械臂已移动到目标位置。")
    
    # 开始定时0.1s采集图像
    start_time = time.time()
    capture_times = []
    while time.time() - start_time <= 2:
        current_time = time.time() - start_time
        capture_times.append(current_time)
        print(f"Captured image at {current_time:.2f} seconds")
        img = mvs.get_image()
        # 在中心位置画一个红色圆圈作为标记
        h, w, _ = img.shape
        cv2.circle(img, (w//2, h//2), 10, (0, 0, 255), -1)
        # 绘制5*5的网格线
        for i in range(1, 5):
            cv2.line(img, (i*w//5, 0), (i*w//5, h), (0, 255, 0), 1)
            cv2.line(img, (0, i*h//5), (w, i*h//5), (0, 255, 0), 1)
        cv2.imwrite(f"{img_dir}capture_{current_time:.2f}.png", img)

    

if __name__ == "__main__":
    
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()