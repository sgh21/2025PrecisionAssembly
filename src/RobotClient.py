#!/usr/bin/env python3
import cv2
import time
import socket
import pickle
import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple
from AuboControlLowLevel import AuboController
from ConstConfig import Const
from Transform import *

# ====== 配置参数 ======
# 通讯协议说明：
# 1. 客户端发送指令为字典，包含以下字段：
#    - 'command': str, 指令类型，如 'detect', 'exit'
#    - 'object': str, 目标对象类型，'hole'或'gear'
#    - 'target_hole_idx': int, 目标圆孔索引，仅当object为'hole'时需要
# 2. 服务端返回结果为字典，包含以下字段：
#     - 'status': str, 'success'或'error'
#     - 'result': 识别结果，若status为'success'，则根据object类型返回不同内容
#     - 'message': str, 错误信息，仅当status为'error'时需要

VISION_HOST = Const.Vision.HOST
VISION_PORT = Const.Vision.PORT

ROBOT_IP = Const.Robot.IP
ROBOT_PORT = Const.Robot.PORT

POS_ERROR_THRESHOLD = Const.Robot.POSE_ERROR_THRESHOLD  # 位置误差阈值（米）

INTRINSIC_U0 = Const.Camera.INTRINSIC_U0
INTRINSIC_A = Const.Camera.INTRINSIC_A
ROBOT_INIT_POS = Const.Robot.INIT_POS
ROBOT_INIT_ORI = Const.Robot.INIT_ORI
HAND_IN_EYE_OFFSET = Const.Robot.HAND_IN_EYE_OFFSET
CONTROLLER_INIT_ANGLE = Const.Robot.CONTROLLER_INIT_ANGLE  

TARGET_HOLE_IDX_LIST = Const.Task.TARGET_HOLE_IDX_LIST
TIME_SLEEP = Const.Task.TIME_SLEEP  # 等待机械臂稳定的时间

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """阻塞读取正好 n 字节；若对端关闭或异常则抛出 EOFError"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf.extend(chunk)
    return bytes(buf)

class RobotClient:
    def __init__(self):
        self.client_socket: Optional[socket.socket] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((VISION_HOST, VISION_PORT))
            print('已连接到服务器')
        except Exception as e:
            print(e)
            print('连接服务器失败')
            self.client_socket = None   

        self.aubo = AuboController(ip=ROBOT_IP, port=ROBOT_PORT, enable_log=True)
        self.set_robot_mode('stable')  # 设置机械臂为稳定模式
# TODO: 此处没问题的话可以接纳
    def send_command(self, command: dict):
        """发送指令"""
        if not self.client_socket:
            raise RuntimeError("未连接到视觉服务器")
        data = pickle.dumps(command)
        data_len = len(data)
        # 先发送4字节长度
        self.client_socket.sendall(data_len.to_bytes(4, byteorder='big'))
        # 再发送数据内容
        self.client_socket.sendall(data)

    def receive_data(self):
        """接收数据，增加异常处理"""
        if not self.client_socket:
            print("未连接到视觉服务器，无法接收数据。")
            return None
        try:
            data_len_bytes = recv_exact(self.client_socket, 4)
            data_len = int.from_bytes(data_len_bytes, byteorder='big')
            # print(f'即将接收数据，长度为：{data_len}字节')
            data_received = recv_exact(self.client_socket, data_len)
            result = pickle.loads(data_received)
            # print('已收到处理后的数据')
            return result
        except (socket.error, pickle.UnpicklingError, EOFError) as e:
            print(f"接收数据时发生异常: {e}")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None
    
    def disconnect(self):
        if self.client_socket:
            try:
                self.client_socket.close()
            finally:
                print('已断开连接')
        self.aubo.disconnect()

    def set_robot_mode(self, mode: str = 'fast'): 
        """设置机械臂运动模式"""
        if mode == 'fast':
            self.aubo.set_joint_maxacc(Const.Robot.JOINT_MAX_ACC)
            self.aubo.set_joint_maxvelc(Const.Robot.JOINT_MAX_VELC)
            self.aubo.set_end_speed(Const.Robot.END_MAX_VELC)
            self.aubo.set_end_acc(Const.Robot.END_MAX_ACC)
        elif mode == 'stable':
            self.aubo.set_joint_maxacc(Const.Robot.JOINT_STABLE_ACC)
            self.aubo.set_joint_maxvelc(Const.Robot.JOINT_STABLE_VELC)
            self.aubo.set_end_speed(Const.Robot.END_STABLE_VELC)
            self.aubo.set_end_acc(Const.Robot.END_STABLE_ACC)
        elif mode == 'insert':
            self.aubo.set_joint_maxacc(Const.Robot.JOINT_INSERT_ACC)
            self.aubo.set_joint_maxvelc(Const.Robot.JOINT_INSERT_VELC)
            self.aubo.set_end_speed(Const.Robot.END_INSERT_VELC)
            self.aubo.set_end_acc(Const.Robot.END_INSERT_ACC)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def move_and_detect(
        self, 
        object: str = 'hole',
        target_hole_idx = Const.Task.TARGET_HOLE_IDX
    ) -> Tuple[List[float], List[float]]:
        """移动机械臂并检测圆孔/齿轮位置"""
        pos_error = 10.0  # 初始位置误差
        current_waypoint = self.aubo.get_current_waypoint()
        current_pos = deepcopy(current_waypoint['pos'])
        current_ori = quaternion_standard2rpy(current_waypoint['ori'])

        while pos_error > POS_ERROR_THRESHOLD:
            # 发送检测命令
            self.send_command({'command': 'detect','object': object, 'target_hole_idx': target_hole_idx})
            result = self.receive_data()
            if not isinstance(result, dict) or result.get('status') != 'success':
                if isinstance(result, dict):
                    print(f"服务端返回错误: {result.get('message', '未知错误')}")
                else:
                    print("服务端返回错误: 未知错误（数据格式异常或未收到数据）")
                continue

            if object == 'hole':
                u, v, r = result['result']
            elif object == 'gear':
                gear_pos, gear_angle = result['result']
                if gear_pos is None:
                    print("Target gear not found")
                    continue
                u, v, r = gear_pos
            else:
                raise ValueError(f"Unsupported object type: {object}")

            # 计算相对移动位置（像素 -> 米，按标定矩阵）
            U = np.array([u, v])
            U0 = np.array(INTRINSIC_U0)
            A = np.array(INTRINSIC_A)
            Delta_X = - np.linalg.inv(A).dot(U - U0)/1000
            # 计算新的末端位置，评估当前检测的误差
            pos_error = np.linalg.norm(Delta_X)
            new_pos = np.array(current_pos) + np.array([Delta_X[0], Delta_X[1], 0])
            new_ori = current_ori  # 姿态保持不变
            print(f"移动到新位置: {new_pos.tolist()}, 姿态: {new_ori}")
            self.aubo.movel(new_pos.tolist(), new_ori, joint=True)
            time.sleep(TIME_SLEEP)  # 等待机械臂稳定
            current_pos = new_pos.tolist()
            current_ori = new_ori

        current_pos = self.aubo.get_current_waypoint()['pos']
        if object == 'hole':
            target_pos = current_pos
            target_ori = quaternion_standard2rpy(self.aubo.get_current_waypoint()['ori'])
            return target_pos, target_ori  # 返回圆孔位置和姿态
        elif object == 'gear':
            target_pos = current_pos  # 齿轮位置不需要偏移
            return target_pos, gear_angle  # 返回齿轮位置和角度

    def reset_and_insert_hole(self, target_hole_idx_list: List[int] = TARGET_HOLE_IDX_LIST):
        """重置机械臂位置并插入圆孔"""
        init_pos = deepcopy(ROBOT_INIT_POS)
        init_ori = deepcopy(ROBOT_INIT_ORI)

        # 移动到初始位置
        self.aubo.movel(init_pos, init_ori, joint=True)
        time.sleep(TIME_SLEEP)  # 等待机械臂稳定

        # 定位齿轮位置和角度
        gear_pos, gear_angle = self.move_and_detect(object='gear')

        print(f"齿轮位置: {gear_pos}, 角度: {gear_angle * 180 / np.pi} deg")

        # 定位圆孔位置
        target_pos_list = []
        target_ori_list = []
        for idx in target_hole_idx_list:
            target_pos, target_ori = self.move_and_detect(object='hole', target_hole_idx=idx)

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
        self,
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
    robot_client = RobotClient()
    print("设备初始化完成。")

    # 如果视觉服务器未连接，直接退出避免后续异常
    if robot_client.client_socket is None:
        print("视觉服务器未连接，退出。")
        robot_client.aubo.disconnect()
        return

    # 获取当前机械臂末端位置和姿态
    target_insert_pos_list, target_insert_ori_list = robot_client.reset_and_insert_hole(TARGET_HOLE_IDX_LIST)
    # 安全点插补
    target_insert_pos_list, target_insert_ori_list = robot_client.interpolation_of_safety_points(
        target_insert_pos_list,
        target_insert_ori_list,
        dz=0.05  # 安全高度偏移，单位：米
    )

    # 插入圆孔
    robot_client.set_robot_mode('insert')  # 设置机械臂为插入模式
    # 插入圆孔
    for pos, ori in zip(target_insert_pos_list, target_insert_ori_list):
        print(f"移动到插入位置: {pos}, 姿态: {ori}")
        robot_client.aubo.movel(pos, ori, joint=False)
        input("请确认圆孔已插入，按回车键继续.")
    print("圆孔插入完成。")

    robot_client.set_robot_mode('fast')  # 设置机械臂为稳定模式
    robot_client.aubo.movel(ROBOT_INIT_POS, ROBOT_INIT_ORI, joint=True)  # 回到初始位置
    print("机械臂已回到初始位置。")
    # 断开连接
    cv2.destroyAllWindows()
    try:
        if robot_client.client_socket is not None:
            robot_client.send_command({'command': 'exit'})  # 发送退出指令
    finally:
        robot_client.disconnect()  # 断开与服务器的连接

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cv2.destroyAllWindows()
        print("齿轮配合任务完成，程序结束。")
