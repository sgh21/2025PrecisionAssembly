import os
import cv2
import time
from copy import deepcopy
from typing import List, Tuple
from AuboControlLowLevel import AuboController
from MVSControl import MVSController
from ComputePose import ImageProcessor
from ConstConfig import Const
from Transform import *

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
CONTROLLER_INIT_ANGLE = Const.Robot.CONTROLLER_INIT_ANGLE  

TIME_SLEEP = Const.Task.TIME_SLEEP  # 等待机械臂稳定的时间

# TODO: 对代码逻辑，尤其是标定时各个方向坐标系变换进行测试
# TODO： 对圆孔识别算法进行进一步优化，测试精度问题的主要诱因 

def move_and_detect(
    aubo_handle: AuboController, 
    mvs_handle: MVSController, 
    img_processor: ImageProcessor,
    object: str = 'hole',
    target_hole_idx: int = Const.Task.TARGET_HOLE_IDX
):
    """移动机械臂并检测圆孔位置"""
    pos_error = 10  # 初始位置误差
    current_waypoint = aubo_handle.get_current_waypoint()
    current_pos = deepcopy(current_waypoint['pos'])
    current_ori = quaternion_standard2rpy(current_waypoint['ori'])
    # 循环拍照并查找齿轮位置和角度
    while pos_error > POS_ERROR_THRESHOLD:
        img = mvs_handle.get_image()
        if img is None:
            print("未获取到图像")
            continue
            
        if object == 'hole':
            hole_list = img_processor.detect_hole(
                img,
                circle_fit_method='EdgeDrawing',
            )

            assert hole_list[target_hole_idx][2] is not None, "Target circle not found"
            u, v, r = hole_list[target_hole_idx]

        elif object == 'gear':
            gear_pos, gear_angle = img_processor.dectect_gear(
                img,
                circle_fit_method='EdgeDrawing',
            )

            assert gear_pos is not None, "Target gear not found"
            u, v, r = gear_pos
        
        else:
            raise ValueError(f"Unsupported object type: {object}")
        
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
        aubo_handle.movel(new_pos.tolist(), new_ori, joint=True)
        time.sleep(TIME_SLEEP)  # 等待机械臂稳定
        current_pos = new_pos.tolist()
        current_ori = new_ori
    
    current_pos = aubo_handle.get_current_waypoint()['pos']
    if object == 'hole':
        target_pos = current_pos
        target_ori = quaternion_standard2rpy(aubo_handle.get_current_waypoint()['ori'])
        return target_pos, target_ori  # 返回圆孔位置和姿态
    elif object == 'gear':
        target_pos = current_pos  # 齿轮位置不需要偏移
        return target_pos, gear_angle  # 返回齿轮位置和角度

def reset_and_insert_hole(aubo_handle: AuboController, 
                          mvs_handle: MVSController, 
                          img_processor: ImageProcessor,
                          target_hole_idx: List[int] = [Const.Task.TARGET_HOLE_IDX]
                          ):
    """重置机械臂位置并插入圆孔
    Args:
        aubo_handle (AuboController): 机械臂控制器实例
        mvs_handle (MVSController): 相机控制器实例
        img_processor (ImageProcessor): 图像处理器实例
        target_hole_idx (List[int]): 目标圆孔索引列表
    """
    init_pos = deepcopy(ROBOT_INIT_POS)
    init_ori = deepcopy(ROBOT_INIT_ORI)

    # 移动到初始位置
    aubo_handle.movel(init_pos, init_ori, joint=True)
    time.sleep(TIME_SLEEP)  # 等待机械臂稳定

    # 定位齿轮位置和角度
    gear_pos, gear_angle = move_and_detect(
        aubo_handle, 
        mvs_handle,
        img_processor,
        object='gear'
    )
    
    print(f"齿轮位置: {gear_pos}, 角度: {gear_angle * 180 / np.pi} deg")

    # 定位圆孔位置
    target_pos_list = []
    target_ori_list = []
    for idx in target_hole_idx:
        target_pos, target_ori = move_and_detect(
            aubo_handle, 
            mvs_handle,
            img_processor,
            object='hole',
            target_hole_idx=idx
        )

        target_pos_list.append(target_pos)
        target_ori_list.append(target_ori)
        print(f"圆孔位置: {target_pos}, 姿态: {target_ori}")

    # 计算中心连线角度
    center_connection_angle_list = []
    delta_controller_angle_list = []
    target_insert_pos_list = []
    target_insert_ori_list = []
    for target_pos in target_pos_list:
        center_connection_angle = np.arctan2(
            target_pos[0] - gear_pos[0], 
            target_pos[1] - gear_pos[1]
        )
        center_connection_angle_list.append(center_connection_angle)
        assert np.abs(( center_connection_angle % (np.pi / 3)) - np.pi / 6) < 5 / 180 * np.pi, '中心连线角度不正确，请检查齿轮和圆孔位置'
        controller_angle = gear_angle2controller_angle(
            gear_angle, 
            center_connection_angle,
            tooth_range=Const.Gear.PRESSURE_ANGLE,
            normal_to_zero=True
        )
        delta_controller_angle = CONTROLLER_INIT_ANGLE - controller_angle
        delta_controller_angle_list.append(delta_controller_angle)
        print(f"中心连线角度: {center_connection_angle * 180 / np.pi} deg")
        print(f"控制器角度: {controller_angle * 180 / np.pi} deg")

        target_insert_pos_list.append(np.array(target_pos) + np.array(HAND_IN_EYE_OFFSET))
        target_insert_ori_list.append(np.array(ROBOT_INIT_ORI) + np.array([0, 0, delta_controller_angle]))

    return target_insert_pos_list, target_insert_ori_list


def interpolation_of_safety_points(
    pos_list: List[List[float]],
    ori_list: List[List[float]],
    dz: float = 0.05  # 安全高度偏移，单位：米
):
    """
    在每个插入点前后插入一个安全点，实现竖直向下插入和竖直向上抬起。
    Args:
        pos_list: 插入点位置列表，每个为[x, y, z]
        ori_list: 插入点姿态列表
        dz: 安全高度偏移量，默认0.05米
    Returns:
        new_pos_list, new_ori_list: 包含安全点的插入路径
    """
    new_pos_list = []
    new_ori_list = []
    for pos, ori in zip(pos_list, ori_list):
        pos = np.array(pos)
        # 插入前的安全点（在z轴上方）
        safe_pos_before = pos.copy()
        safe_pos_before[2] += dz
        new_pos_list.append(safe_pos_before.tolist())
        new_ori_list.append(ori)
        # 插入点
        new_pos_list.append(pos.tolist())
        new_ori_list.append(ori)
        # 抬起后的安全点（在z轴上方）
        safe_pos_after = pos.copy()
        safe_pos_after[2] += dz
        new_pos_list.append(safe_pos_after.tolist())
        new_ori_list.append(ori)
    return new_pos_list, new_ori_list

def main():
    # 初始化机器人和相机
    aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT, enable_log=True)
    mvs = MVSController()
    img_processor = ImageProcessor(model_weights=YOLO_WEIGHTS, show=True, waitkey=100)
    aubo.set_joint_maxacc(JOINT_MAX_ACC)
    aubo.set_joint_maxvelc(JOINT_MAX_VELC)
    aubo.set_end_speed(END_MAX_VELC)
    aubo.set_end_acc(END_MAX_ACC)

    print("设备初始化完成。")

    # 获取当前机械臂末端位置和姿态
    target_insert_pos_list, target_insert_ori_list = reset_and_insert_hole(
        aubo, 
        mvs, 
        img_processor,
        target_hole_idx=[0,1,2,3,4,5]  # 假设有三个目标圆孔
    )
    # 安全点插补
    target_insert_pos_list, target_insert_ori_list = interpolation_of_safety_points(
        target_insert_pos_list, 
        target_insert_ori_list,
        dz=0.05  # 安全高度偏移，单位：米
    )

    # 插入圆孔
    aubo.set_joint_maxacc(Const.Robot.JOINT_INSERT_ACC)
    aubo.set_joint_maxvelc(Const.Robot.JOINT_INSERT_VELC)
    aubo.set_end_speed(Const.Robot.END_INSERT_VELC)
    aubo.set_end_acc(Const.Robot.END_INSERT_ACC)
    for pos, ori in zip(target_insert_pos_list, target_insert_ori_list):
        print(f"移动到插入位置: {pos}, 姿态: {ori}")
        aubo.movel(pos, ori, joint=False)
        input("请确认圆孔已插入，按回车键继续...")
    print("圆孔插入完成。")

    aubo.movel(ROBOT_INIT_POS, ROBOT_INIT_ORI, joint=True)  # 回到初始位置
    print("机械臂已回到初始位置。")
    # 断开连接
    cv2.destroyAllWindows()
    aubo.disconnect()
    mvs.close_device()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cv2.destroyAllWindows()
        print("手眼标定完成，程序结束。")