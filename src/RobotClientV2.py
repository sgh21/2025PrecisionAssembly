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
TARGET_HOLE_IDX = Const.Task.TARGET_HOLE_IDX  # 默认目标圆孔索引
TIME_SLEEP = Const.Task.TIME_SLEEP  # 等待机械臂稳定的时间
STEP = Const.Robot.STEP
DZ = Const.Robot.DZ

#CLASS INFO
GEAR = Const.ClassInfo.GEAR_CLASS
HOLE = Const.ClassInfo.HOLE_CLASS

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

    @staticmethod
    def decode_img(img_bytes):
        """将基础格式（bytes）解码为 numpy 图像"""
        if img_bytes is None:
            return None
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

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
    
    def detect(self, 
               object: str, 
               target_hole_idx: int = Const.Task.TARGET_HOLE_IDX) -> Optional[Tuple[float, float, float]]:
        """检测指定物体的位置和姿态"""
        self.send_command({'command': 'detect', 'object': object, 'target_hole_idx': target_hole_idx})
        result = self.receive_data()
        if not isinstance(result, dict) or result.get('status') != 'success':
            if isinstance(result, dict):
                print(f"服务端返回错误: {result.get('message', '未知错误')}")
            else:
                print("服务端返回错误: 未知错误（数据格式异常或未收到数据）")
        img_bytes = result.get('img')
        img = self.decode_img(img_bytes)
        return result.get('result'), img

    def move_and_detect(
        self, 
        object: str = HOLE,
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

            if object == HOLE:
                u, v, r = result['result']
            elif object == GEAR:
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
        if object == HOLE:
            target_pos = current_pos
            target_ori = quaternion_standard2rpy(self.aubo.get_current_waypoint()['ori'])
            return target_pos, target_ori  # 返回圆孔位置和姿态
        elif object == GEAR:
            target_pos = current_pos  # 齿轮位置不需要偏移
            return target_pos, gear_angle  # 返回齿轮位置和角度

    def calculate_safe_approach_waypoints(self, 
                                       gear_pos, 
                                       gear_angle,
                                       target_pos, 
                                       target_ori, 
                                       step=0.005, 
                                       dz=0.05):
        """
        插孔动作分步：先到连线方向更远处，再靠近孔口，最后竖直插入
        gear_pos: 齿轮中心 [x, y, z]
        gear_angle: 齿轮角度
        target_pos: 圆孔中心 [x, y, z]
        target_ori: 插入姿态
        step: 预插入距离（米）
        dz: 安全高度偏移（米）
        """

        center_connection_angle = np.arctan2(
                target_pos[0] - gear_pos[0], 
                target_pos[1] - gear_pos[1]
        )

        assert np.abs(( center_connection_angle % (np.pi / 3)) - np.pi / 6) < 5 / 180 * np.pi, '中心连线角度不正确，请检查齿轮和圆孔位置'
        
        controller_angle = gear_angle2controller_angle(
                gear_angle, 
                center_connection_angle,
                tooth_range=Const.Gear.PRESSURE_ANGLE,
                normal_to_zero=True
        )

        delta_controller_angle = CONTROLLER_INIT_ANGLE - controller_angle
        target_insert_pos = np.array(target_pos) +np.array(HAND_IN_EYE_OFFSET) 
        target_insert_ori = np.array(target_ori) + np.array([0, 0, delta_controller_angle])

        target_insert_pos_list = []
        target_insert_ori_list = []

        # 计算连线方向单位向量
        target_pos_np = np.array(target_pos)
        gear_pos_np = np.array(gear_pos)
        # 机器人坐标系下的direction
        direction = target_pos_np[:2] - gear_pos_np[:2]
        direction = direction / np.linalg.norm(direction)

        # 预插入点（在连线方向更远 step）
        pre_pos_xy = target_insert_pos[:2] + direction * step
        pre_pos = np.array([pre_pos_xy[0], pre_pos_xy[1], target_insert_pos[2] + dz])

        # 啮合点（靠近孔口但不进入，距离 step/2）
        engage_pos_xy = target_insert_pos[:2]
        engage_pos = np.array([engage_pos_xy[0], engage_pos_xy[1], target_insert_pos[2] + dz / 2.0])

        # 插入点（竖直向下到孔口）
        insert_pos = deepcopy(target_insert_pos)

        # 插入点下方安全点
        safe_pos = np.array([target_insert_pos[0], target_insert_pos[1], target_insert_pos[2] + dz])

        # 将所有点添加到列表中
        target_insert_pos_list.extend([pre_pos.tolist(), engage_pos.tolist(), insert_pos.tolist(), safe_pos.tolist()])
        target_insert_ori_list.extend([target_insert_ori] * 4)

        return target_insert_pos_list, target_insert_ori_list
    
    def reset_and_insert_hole(self, 
                              target_hole_idx: List[int] = TARGET_HOLE_IDX, 
                              gear_pos: List[float] = None, 
                              gear_angle: float = None
                              ) -> Tuple[List[List[float]], List[List[float]]]:
        """重置机械臂位置并插入圆孔"""
        init_pos = deepcopy(ROBOT_INIT_POS)
        init_ori = deepcopy(ROBOT_INIT_ORI)

        # 移动到初始位置
        self.set_robot_mode('stable')
        self.aubo.movel(init_pos, init_ori, joint=True)
        time.sleep(TIME_SLEEP)  # 等待机械臂稳定

        if gear_pos is None or gear_angle is None:
            gear_pos, gear_angle = self.move_and_detect(object=GEAR)

        print(f"齿轮位置: {gear_pos}, 角度: {gear_angle * 180 / np.pi} deg")

        target_pos, target_ori = self.move_and_detect(object=HOLE, target_hole_idx=target_hole_idx)
        print(f"目标圆孔位置: {target_pos}, 姿态: {target_ori}")

        target_insert_pos_list, target_insert_ori_list = self.calculate_safe_approach_waypoints(
            gear_pos=gear_pos,
            gear_angle=gear_angle,
            target_pos=target_pos,
            target_ori=target_ori,
            step = STEP,
            dz = DZ
        )

        self.set_robot_mode('insert')  # 设置机械臂为插入模式
        for pos, ori in zip(target_insert_pos_list, target_insert_ori_list):
            print(f"插入位置: {pos}, 姿态: {ori}")
            self.aubo.movel(pos, ori, joint=False)
            # input("请确认圆孔已插入，按回车键继续.")
        else:
            print("圆孔插入完成，回到初始位置。")
            self.set_robot_mode('fast')
            self.aubo.movel(init_pos, init_ori, joint=True)

        return gear_pos, gear_angle

def main():
    # 初始化机器人和相机
    robot_client = RobotClient()
    print("设备初始化完成。")

    # 如果视觉服务器未连接，直接退出避免后续异常
    if robot_client.client_socket is None:
        print("视觉服务器未连接，退出。")
        robot_client.aubo.disconnect()
        return
    gear_pos , gear_angle, gear_flag_str = None, None, 'n'
    for target_hole_idx in TARGET_HOLE_IDX_LIST:
        print(f"开始处理目标圆孔索引: {target_hole_idx}")
        gear_flag = gear_flag_str == 'y'
        gear_pos, gear_angle = robot_client.reset_and_insert_hole(
            target_hole_idx=target_hole_idx,
            gear_pos=gear_pos if gear_flag else None,
            gear_angle=gear_angle if gear_flag else None
        )
        gear_flag_str = input("请确认是否使用上次检测齿轮位置和角度(y/n): ").strip().lower()

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
