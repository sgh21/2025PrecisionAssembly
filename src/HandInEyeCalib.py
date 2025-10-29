#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
from copy import deepcopy
import os, sys
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("workspace:", workspace)
sys.path.append(workspace)
sys.path.append(os.path.join(workspace, 'configs'))
from RobotClientV2 import RobotClient
from ConstConfig import Const
from Transform import *

"""
基于客户端-服务端架构的手眼标定程序
在机器人移动到目标孔位置后，使用此程序将机器人和目标孔位进行对中
获得机器人的相对手眼和齿轮的标定角度
"""

# ====== 配置参数 ======
X_OFFSET = Const.Robot.X_OFFSET  # 相机X轴偏移量（米）
ROBOT_INIT_POS = Const.Robot.INIT_POS
ROBOT_INIT_ORI = Const.Robot.INIT_ORI

INTRINSIC_U0 = Const.Camera.INTRINSIC_U0
INTRINSIC_A = Const.Camera.INTRINSIC_A

TIME_SLEEP = Const.Task.TIME_SLEEP  # 等待机械臂稳定的时间
TARGET_HOLE_IDX = Const.Task.TARGET_HOLE_IDX

# CLASS INFO
GEAR = Const.ClassInfo.GEAR_CLASS
HOLE = Const.ClassInfo.HOLE_CLASS

class HandInEyeCalibClient(RobotClient):
    """继承RobotClient，专门用于手眼标定操作"""
    
    def __init__(self):
        super().__init__()
        print("手眼标定客户端初始化完成")
    
    def calibrate_hole_position(self, target_hole_idx: int = TARGET_HOLE_IDX):
        """标定圆孔位置，返回位置偏移"""
        print(f"开始标定目标圆孔索引: {target_hole_idx}")
        
        # 检查视觉服务器连接
        if self.client_socket is None:
            print("视觉服务器未连接，无法执行手眼标定")
            return None
        
        try:
            # 获取当前机械臂末端位置和姿态
            current_waypoint = self.aubo.get_current_waypoint()
            insert_pos = deepcopy(current_waypoint['pos'])
            init_pos = current_waypoint['pos']
            init_ori = quaternion_standard2rpy(current_waypoint['ori'])
            
            # 移动到初始位置
            self.set_robot_mode('stable')
            init_pos[2] = ROBOT_INIT_POS[2]  # 确保Z轴位置正确
            self.aubo.movel(init_pos, init_ori)
            init_pos[0] += X_OFFSET  # 假设相机在机械臂末端前方0.11米
            self.aubo.movel(init_pos, init_ori, joint=True)

            time.sleep(TIME_SLEEP)
            
            # 使用父类的move_and_detect方法精确定位到圆孔
            target_pos, target_ori = self.move_and_detect(
                object=HOLE, 
                target_hole_idx=target_hole_idx
            )
            
            print(f"检测到目标圆孔位置: {target_pos}")
            
            # 计算位置偏移
            current_pos = self.aubo.get_current_waypoint()['pos']
            hole_pos = deepcopy(current_pos)
            pos_offset = np.array(insert_pos) - np.array(current_pos)
            
            print(f"\033[32m手眼标定结果（位置偏移）: {pos_offset} 米\033[0m")
            
            return pos_offset, hole_pos
            
        except Exception as e:
            print(f"圆孔位置标定时发生错误: {e}")
            return None
    
    def calibrate_gear_angle(self, hole_pos):
        """标定齿轮角度，返回控制器角度"""
        print("开始标定齿轮角度")
        
        try:
            # 移动到齿轮keyhole位置进行齿轮角度测量
            self.aubo.movel(pos=ROBOT_INIT_POS, ori=ROBOT_INIT_ORI, joint=True)
            time.sleep(TIME_SLEEP)
            
            # 使用父类的move_and_detect方法精确定位到齿轮
            gear_pos, gear_angle = self.move_and_detect(object=GEAR, times_limit=1)
            
            print(f"检测到齿轮位置: {gear_pos}, 角度: {gear_angle * 180 / np.pi:.2f} deg")
            
            # 获取当前位置作为齿轮keyhole位置
            current_pos = self.aubo.get_current_waypoint()['pos']
            keyhole_pos = deepcopy(current_pos)
            
            # 计算中心连线角度
            # 机器人坐标系和相机坐标系x,y轴相反，理论值应该是 -30
            center_connection_angle = np.arctan2(
                hole_pos[0] - keyhole_pos[0], 
                hole_pos[1] - keyhole_pos[1]
            )
            
            assert np.abs(( center_connection_angle % (np.pi / 3)) - np.pi / 6) < 5 / 180 * np.pi, '中心连线角度不正确，请检查齿轮和圆孔位置'
            print(f"\033[33m中心连线角度: {center_connection_angle * 180 / np.pi:.2f} deg\033[0m")
            
            # 计算控制器角度
            controller_angle = gear_angle2controller_angle(
                gear_angle, 
                center_connection_angle,
                tooth_range=Const.Gear.PRESSURE_ANGLE,
                normal_to_zero=True
            )
            
            controller_angle_deg = controller_angle * 180 / np.pi
            print(f"\033[32m控制器角度: {controller_angle_deg:.2f} deg\033[0m")
            
            return controller_angle, gear_angle, center_connection_angle
            
        except Exception as e:
            print(f"齿轮角度标定时发生错误: {e}")
            return None
    
    def perform_hand_in_eye_calibration(self, target_hole_idx: int = TARGET_HOLE_IDX, pass_gear: bool = False):
        """执行完整的手眼标定流程"""
        print("开始手眼标定流程...")
        
        # 第一步：标定圆孔位置
        hole_result = self.calibrate_hole_position(target_hole_idx)
        if hole_result is None:
            print("圆孔位置标定失败")
            return False
            
        pos_offset, hole_pos = hole_result
        
        # 第二步：标定齿轮角度（可选）
        if not pass_gear:
            gear_result = self.calibrate_gear_angle(hole_pos)
            if gear_result is None:
                print("齿轮角度标定失败")
                return False
            
            controller_angle, gear_angle, center_connection_angle = gear_result
            
            print(f"\n\033[32m=== 手眼标定完成 ===\033[0m")
            print(f"位置偏移: {pos_offset} 米")
            print(f"齿轮角度: {gear_angle * 180 / np.pi:.2f} deg")
            print(f"中心连线角度: {center_connection_angle * 180 / np.pi:.2f} deg")
            print(f"控制器角度: {controller_angle * 180 / np.pi:.2f} deg")
        else:
            print(f"\n\033[32m=== 手眼标定完成（跳过齿轮） ===\033[0m")
            print(f"位置偏移: {pos_offset} 米")
        
        return True
    
    def return_to_init_position(self):
        """返回到初始位置"""
        try:
            self.set_robot_mode('fast')
            self.aubo.movel(pos=ROBOT_INIT_POS, ori=ROBOT_INIT_ORI, joint=True)
            time.sleep(TIME_SLEEP)
            print("已返回初始位置")
        except Exception as e:
            print(f"返回初始位置时发生错误: {e}")

def main():
    """主函数"""
    print("初始化手眼标定客户端...")
    
    # 创建手眼标定客户端（继承自RobotClient）
    calib_client = HandInEyeCalibClient()
    
    real_hole_pos = calib_client.aubo.get_current_waypoint()['pos']

    # 检查连接状态
    if calib_client.client_socket is None:
        print("视觉服务器未连接，退出程序")
        calib_client.aubo.disconnect()
        return
    
    print("设备初始化完成")
    
    try:
        # 获取用户输入
        target_hole_idx = TARGET_HOLE_IDX
        user_input = input(f"请输入目标圆孔索引 (默认: {target_hole_idx}): ").strip()
        if user_input:
            try:
                target_hole_idx = int(user_input)
            except ValueError:
                print("输入无效，使用默认值")
        
        pass_gear_input = input("是否跳过齿轮角度检测？(y/n): ").strip().lower()
        pass_gear = True if pass_gear_input == 'y' else False
        
        # 执行手眼标定
        success = calib_client.perform_hand_in_eye_calibration(
            target_hole_idx=target_hole_idx, 
            pass_gear=pass_gear
        )
        
        if success:
            print("手眼标定完成！")
        else:
            print("手眼标定失败")
        
        # 返回初始位置
        calib_client.return_to_init_position()
        
    except KeyboardInterrupt:
        print("收到中断信号，程序退出")
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
    finally:
        real_hole_pos[2] = ROBOT_INIT_POS[2]  # 确保Z轴位置正确
        calib_client.aubo.movel(pos=real_hole_pos, ori=Const.Robot.INIT_ORI)
        # 清理资源（继承自RobotClient的disconnect方法）
        calib_client.disconnect()
        cv2.destroyAllWindows()
        print("手眼标定程序结束")

if __name__ == "__main__":
    main()