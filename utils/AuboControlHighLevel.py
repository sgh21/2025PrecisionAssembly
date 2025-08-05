from AuboControlLowLevel import AuboController
from config import *
from Transform import quaternion_standard2rpy

def movel_step(aubo_handle, axis, step=0.025):
    """
    沿指定轴方向步进移动末端
    :param aubo_handle: Aubo控制器实例
    :param axis: 指定轴，0: x, 1: y, 2: z
    :param step: 步长（单位：米）
    """
    current_waypoint = aubo_handle.get_current_waypoint()
    current_pos = current_waypoint['pos']
    current_ori = quaternion_standard2rpy(current_waypoint['ori'])
    new_pos = [current_pos[i] + (step / 1000 if i == axis else 0) for i in range(3)]
    aubo_handle.movel(new_pos, current_ori, joint=True)
    print(f"Moved axis {axis} by {step} mm. New pos: {new_pos}")

def main():
    # 连接机械臂
    aubo = AuboController()
    print("AuboController initialized.")
    aubo.set_joint_maxacc(JOINT_MAX_ACC)
    aubo.set_joint_maxvelc(JOINT_MAX_VELC)
    aubo.set_end_speed(END_MAX_VELC)
    aubo.set_end_acc(END_MAX_ACC)
    print("WASDQE控制末端步进（小写0.025mm，大写0.05mm），q/e为z轴正/负，按ESC或Ctrl+C退出。")
    key_map = {
        'w': (0, 0.025), 'W': (0, 0.05),  # x+
        's': (0, -0.025), 'S': (0, -0.05), # x-
        'a': (1, 0.025), 'A': (1, 0.05),  # y+
        'd': (1, -0.05), 'D': (1, -0.05), # y-
        'q': (2, 0.025), 'Q': (2, 0.05),  # z+
        'e': (2, -0.025), 'E': (2, -0.05) # z-
    }
    try:
        while True:
            key = input("请输入步进指令(WASDQE/wadsqe)：").strip()
            if key in key_map:
                axis, step = key_map[key]
                movel_step(aubo, axis, step)
            elif key.lower() == 'exit' or key == '\x1b':  # ESC或exit退出
                print("退出步进模式。")
                break
            else:
                print("无效输入，请输入WASDQE/wadsqe或exit退出。")
    except KeyboardInterrupt:
        print("\n用户中断，退出步进模式。")
    finally:
        aubo.disconnect()

if __name__ == "__main__":
    main()