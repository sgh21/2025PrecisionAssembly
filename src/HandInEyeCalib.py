import os
import cv2
import time
from copy import deepcopy
from AuboControlLowLevel import AuboController
from MVSControl import MVSController
from ComputePose import ImageProcessor
from ConstConfig import Const
from Transform import *

"""
在机器人移动到目标孔位置后，使用此程序将机器人和目标孔位进行对中
获得机器人的相对手眼和齿轮的标定角度
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

TIME_SLEEP = Const.Task.TIME_SLEEP  # 等待机械臂稳定的时间
TARGET_HOLE_IDX = Const.Task.TARGET_HOLE_IDX

def hand_in_eye_calibration(pass_gear = False):
    # 初始化机器人和相机
    aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT, enable_log=True)
    mvs = MVSController()
    img_processor = ImageProcessor(model_weights=YOLO_WEIGHTS, show=True)
    aubo.set_joint_maxacc(JOINT_MAX_ACC)
    aubo.set_joint_maxvelc(JOINT_MAX_VELC)
    aubo.set_end_speed(END_MAX_VELC)
    aubo.set_end_acc(END_MAX_ACC)
    print("设备初始化完成。")

    # 获取当前机械臂末端位置和姿态
    current_waypoint = aubo.get_current_waypoint()
    insert_pos = deepcopy(current_waypoint['pos'])
    init_pos = current_waypoint['pos']
    init_ori = quaternion_standard2rpy(current_waypoint['ori'])
    init_pos[0] += X_OFFSET  # 假设相机在机械臂末端前方0.11米
    init_pos[2] = ROBOT_INIT_POS[2]  # 确保Z轴位置正确
    

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
        hole_list, _ = img_processor.detect_hole(
            img,
            circle_fit_method='EdgeDrawing',
        )
        idx = TARGET_HOLE_IDX
        assert hole_list[idx][2] is not None, "Target circle not found"
        u, v, r = hole_list[idx]
        
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
    hole_pos = deepcopy(current_pos)
    pos_offset = np.array(insert_pos) - np.array(current_pos)
    print(f"最终位置偏移: {pos_offset} 米")
    
    if not pass_gear:
        # 移动到齿轮keyhole位置进行齿轮角度测量
        aubo.movel(pos = ROBOT_INIT_POS, ori = ROBOT_INIT_ORI, joint=True)
        current_pos = aubo.get_current_waypoint()['pos']
        pos_error = 10  # 重置位置误差
        while pos_error > POS_ERROR_THRESHOLD :
            img = mvs.get_image()
            if img is None:
                print("未获取到图像")
                continue
            gear_pos, gear_angle, _ = img_processor.detect_gear(
                img,
                circle_fit_method='EdgeDrawing',
            )

            assert gear_pos is not None, "未检测到齿轮"
            u, v, r = gear_pos
            
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
        keyhole_pos = deepcopy(current_pos)
        # *： 机器人坐标系和相机坐标系x,y轴相反， 理论值应该是 - 30
        center_connection_angle = np.arctan2(hole_pos[0] - keyhole_pos[0], hole_pos[1] - keyhole_pos[1])
        print(f"\033[33m中心连线角度: {center_connection_angle * 180 / np.pi} deg\033[0m")
        # assert np.abs(center_connection_angle % np.pi / 3 - np.pi / 6) < 5 / 180 * np.pi, "中心连线角度异常"
        gear_final_pos = gear_pos
        gear_final_angle = gear_angle
        gear_final_angle_deg = gear_final_angle * 180 / np.pi
        print(f"齿轮位置: {gear_final_pos}, 角度: {gear_final_angle_deg} deg")
        controller_angle = gear_angle2controller_angle(
            gear_final_angle, 
            center_connection_angle,
            tooth_range=Const.Gear.PRESSURE_ANGLE,
            normal_to_zero=True
        )
        controller_angle_deg = controller_angle * 180 / np.pi
        print(f"控制器角度: {controller_angle_deg} deg")
        print(f"手眼标定结果：{pos_offset} 米")
    aubo.movel(pos = ROBOT_INIT_POS, ori = ROBOT_INIT_ORI, joint=True)
    aubo.disconnect()
    mvs.close_device()

if __name__ == "__main__":
    try:
        pass_gear = input("是否跳过齿轮角度检测？(y/n): ").strip().lower()
        pass_gear = True if pass_gear == 'y' else False

        hand_in_eye_calibration()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cv2.destroyAllWindows()
        print("手眼标定完成，程序结束。")