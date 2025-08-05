import os
import cv2
import time
from copy import deepcopy
from AuboControlLowLevel import AuboController
from MVSControl import MVSController
from ConstConfig import Const
from ComputePose import ImageProcessor
from Transform import *

"""
根据Target Hole Index进行插孔，用于辅助手眼标定，将机器人移动到大概位置
需要注意的是，插孔时为了防止碰撞，请将安装板上的齿轮去除，防止碰撞
"""
# ====== 配置参数 ======
ROBOT_IP = Const.Robot.IP
ROBOT_PORT = Const.Robot.PORT
X_OFFSET = Const.Robot.X_OFFSET  # 相机X轴偏移量（米）
POS_ERROR_THRESHOLD = Const.Robot.POSE_ERROR_THRESHOLD  # 位置误差阈值（米） 0.03mm
JOINT_MAX_ACC = Const.Robot.JOINT_MAX_ACC
JOINT_MAX_VELC = Const.Robot.JOINT_MAX_VELC
END_MAX_ACC = Const.Robot.END_MAX_ACC
END_MAX_VELC = Const.Robot.END_MAX_VELC
ROBOT_INIT_POS = Const.Robot.INIT_POS
ROBOT_INIT_ORI = Const.Robot.INIT_ORI
YOLO_WEIGHTS = os.path.join(Const.Yolo.MODEL_DIE, 
                            Const.Yolo.YOLO_HOLE_WEIGHTS)

INTRINSIC_U0 = Const.Camera.INTRINSIC_U0
INTRINSIC_A = Const.Camera.INTRINSIC_A
HAND_IN_EYE_OFFSET = Const.Robot.HAND_IN_EYE_OFFSET

TIME_SLEEP =  Const.Task.TIME_SLEEP  # 等待机械臂稳定的时间

def main():
    # 初始化机器人和相机
    img_processor = ImageProcessor(model_weights=YOLO_WEIGHTS, show=True)
    aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT, enable_log=True)
    mvs = MVSController()
    aubo.set_joint_maxacc(JOINT_MAX_ACC)
    aubo.set_joint_maxvelc(JOINT_MAX_VELC)
    aubo.set_end_speed(END_MAX_VELC)
    aubo.set_end_acc(END_MAX_ACC)
    init_pos = deepcopy(ROBOT_INIT_POS)
    init_ori = deepcopy(ROBOT_INIT_ORI)
    print("设备初始化完成。")

    # 移动到初始位置
    aubo.movel(init_pos, init_ori, joint=True)
    time.sleep(TIME_SLEEP)  # 等待机械臂稳定

    # 循环拍照并查找圆
    pos_error = 10 # 位置误差
    current_pos = deepcopy(init_pos)
    current_ori = deepcopy(init_ori)
    while pos_error > POS_ERROR_THRESHOLD:
        img = mvs.get_image()
        if img is None:
            print("未获取到图像")
            continue
        
        hole_list = img_processor.detect_hole(
            img,
            circle_fit_method='EdgeDrawing',
        )
        idx = Const.Task.TARGET_HOLE_IDX
        assert hole_list[idx][2] is not None, "Target circle not found"
        u, v, r = hole_list[idx]
        print(f"圆心: ({u:.3f}, {v:.3f}), 半径: {r:.3f}")
        
        # 计算相对移动位置
        U = np.array([u, v])
        U0 = np.array(INTRINSIC_U0)
        A = np.array(INTRINSIC_A)
        Delta_X = - np.linalg.inv(A).dot(U - U0)/1000
        
        # 计算新的末端位置，评估当前检测的误差
        pos_error = np.linalg.norm(Delta_X)
        new_pos = np.array(current_pos) + np.array([Delta_X[0], Delta_X[1], 0])
        new_ori = current_ori  # 姿态保持不变
        print(f"移动到新位置: {current_pos}, 姿态: {current_ori}")
        aubo.movel(new_pos.tolist(), new_ori, joint=True)
        time.sleep(TIME_SLEEP)  # 等待机械臂稳定
        current_pos = new_pos.tolist()
        current_ori = new_ori

    current_pos = aubo.get_current_waypoint()['pos']
    target_pos = np.array(current_pos) + np.array(HAND_IN_EYE_OFFSET)
    target_ori = quaternion_standard2rpy(aubo.get_current_waypoint()['ori'])
    aubo.movel(target_pos.tolist(), target_ori, joint=True)  # 移动到手眼标定位置
    time.sleep(TIME_SLEEP)  # 等待机械臂稳定
    aubo.disconnect()
    mvs.close_device()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cv2.destroyAllWindows()
        print("已经移动到目标孔位置，程序结束。")