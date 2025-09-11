import os
import cv2
import time
from copy import deepcopy
import os, sys
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("workspace:", workspace)
sys.path.append(workspace)
sys.path.append(os.path.join(workspace, 'configs'))
from ConstConfig import Const
from RobotClientV2 import RobotClient
from Transform import *

"""
根据Target Hole Index进行插孔，用于辅助手眼标定，将机器人移动到大概位置
需要注意的是，插孔时为了防止碰撞，请将安装板上的齿轮去除，防止碰撞
"""
#!/usr/bin/env python3
import time
from copy import deepcopy
from RobotClientV2 import RobotClient
from ConstConfig import Const

"""
基于RobotClient的插孔程序，用于辅助手眼标定
将机器人移动到指定孔位的大概位置
注意：插孔时请将安装板上的齿轮去除，防止碰撞
"""

# ====== 配置参数 ======
TIME_SLEEP = Const.Task.TIME_SLEEP
TARGET_HOLE_IDX = Const.Task.TARGET_HOLE_IDX
ROBOT_INIT_POS = Const.Robot.INIT_POS
ROBOT_INIT_ORI = Const.Robot.INIT_ORI
HAND_IN_EYE_OFFSET = Const.Robot.HAND_IN_EYE_OFFSET

DZ = Const.Robot.DZ  # 插入时的Z轴偏移量
# CLASS INFO
HOLE = Const.ClassInfo.HOLE_CLASS

class HoleInsertClient(RobotClient):
    """继承RobotClient，专门用于插孔操作"""
    
    def __init__(self):
        super().__init__()
        print("插孔客户端初始化完成")
    
    def move_to_hole_position(self, target_hole_idx: int = TARGET_HOLE_IDX):
        """移动机械臂到指定圆孔位置"""
        print(f"开始移动到目标圆孔索引: {target_hole_idx}")
        
        # 检查视觉服务器连接
        if self.client_socket is None:
            print("视觉服务器未连接，无法执行插孔操作")
            return False
        
        try:
            # 移动到初始位置
            init_pos = deepcopy(ROBOT_INIT_POS)
            init_ori = deepcopy(ROBOT_INIT_ORI)
            self.set_robot_mode('stable')
            self.aubo.movel(init_pos, init_ori, joint=True)
            time.sleep(TIME_SLEEP)
            
            # 使用父类的move_and_detect方法移动到目标位置
            target_pos, target_ori = self.move_and_detect(
                object=HOLE, 
                target_hole_idx=target_hole_idx
            )
            
            print(f"检测到目标圆孔位置: {target_pos}, 姿态: {target_ori}")
            
            # 移动到手眼标定位置（添加偏移）
            current_pos = self.aubo.get_current_waypoint()['pos']
            final_pos = [
                current_pos[0] + HAND_IN_EYE_OFFSET[0],
                current_pos[1] + HAND_IN_EYE_OFFSET[1], 
                current_pos[2] + HAND_IN_EYE_OFFSET[2] + DZ
            ]
            final_ori = quaternion_standard2rpy(self.aubo.get_current_waypoint()['ori'])
            
            print(f"移动到最终插入位置: {final_pos}")
            self.aubo.movel(final_pos, final_ori, joint=True)
            time.sleep(TIME_SLEEP)
            
            print(f"已成功移动到目标圆孔位置 (索引: {target_hole_idx})")
            return True
            
        except Exception as e:
            print(f"移动到圆孔位置时发生错误: {e}")
            return False

def main():
    """主函数"""
    print("初始化插孔客户端...")
    
    # 创建插孔客户端（继承自RobotClient）
    hole_client = HoleInsertClient()
    
    # 检查连接状态
    if hole_client.client_socket is None:
        print("视觉服务器未连接，退出程序")
        hole_client.aubo.disconnect()
        return
    
    print("设备初始化完成")
    
    try:
        # 获取目标孔位索引
        target_hole_idx = TARGET_HOLE_IDX
        user_input = input(f"请输入目标圆孔索引 (默认: {target_hole_idx}): ").strip()
        if user_input:
            try:
                target_hole_idx = int(user_input)
            except ValueError:
                print("输入无效，使用默认值")
        
        # 执行插孔操作
        success = hole_client.move_to_hole_position(target_hole_idx)
        
        if success:
            print("插孔定位完成！")
        else:
            print("插孔操作失败")
        
    except KeyboardInterrupt:
        print("收到中断信号，程序退出")
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
    finally:
        # 清理资源（继承自RobotClient的disconnect方法）
        hole_client.disconnect()
        print("插孔程序结束")

if __name__ == "__main__":
    main()