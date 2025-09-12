import os, sys
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(workspace, 'configs'))

from Aubo5i.robotcontrol import *
from Transform import *
import copy

class AuboController:
    def __init__(self, ip='192.168.70.100', port=8899, enable_log=True):
        self.enable_log = enable_log 
        if self.enable_log:
            # 初始化logger
            logger_init()
            # 启动测试
            logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))
        # 系统初始化
        Auboi5Robot.initialize()
        # 创建机械臂控制类
        self.robot = Auboi5Robot()

        # 创建上下文
        self.handle = self.robot.create_context()

        if self.enable_log:
            # 打印日志
            logger.info("AuboController initialized.")
            logger.info("robot handle={0}".format(self.handle))
            
            self.queue = Queue()
            self.p = Process(target=runWaypoint, args=(self.queue,))
            self.p.start()
            print("Logger process started.")
        else:
            self.queue = None
            self.p = None
        
        # 连接机器人
        self.connect(ip, port)

    def connect(self, ip, port):
        try:
            result =self.robot.connect(ip, port)
            print(result)
            if result != RobotErrorType.RobotError_SUCC:
                if self.enable_log:
                    logger.info("connect to server{0}:{1} failed.".format(ip, port))
                else:
                    print("connect to server{0}:{1} failed.".format(ip, port))
            else:
                if self.enable_log:
                    logger.info("connect to server{0}:{1} successed.".format(ip, port))
                else:
                    print("connect to server{0}:{1} successed.".format(ip, port))
                self.robot.enable_robot_event()
                self.robot.init_profile()
                
                self.robot.set_end_max_line_acc(0.1)
                self.robot.set_end_max_line_velc(0.12)

        except KeyboardInterrupt:
            self.robot.move_stop()

        except RobotError as e:
            if self.enable_log:
                logger.error("robot Event:{0}".format(e))
            else:
                print("robot Event:{0}".format(e))

    def disconnect(self):
        if self.robot.connected:
            # 断开机械臂链接
            self.robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        # print("run end-------------------------")
        if self.enable_log and self.queue is not None:
            self.queue.put('quit')

    def set_tool_kinematics_param(self,tool_kinematics_param):
        """
        Fuck!
        函数没球用
        """
        return self.robot.set_tool_kinematics_param(tool_kinematics_param)
    
    def is_safe(self, current_joint, target_joint, max_diff_threshold = 90):
        if current_joint is None or target_joint is None:
            return False
        
        # 深拷贝输入数据，防止对原始数据进行修改
        current = np.array(copy.deepcopy(current_joint))
        target = np.array(copy.deepcopy(target_joint))
        # print("current_joint ",current * 180/np.pi)
        # print('target_joint', target * 180 / np.pi)
        joint_diff = np.abs(target - current)*180 / np.pi

        max_diff = np.max(joint_diff)
        
        return max_diff <= max_diff_threshold
    
    def movej(self, joint_radian):
        joint_radian = tuple(joint_radian)
    
        if self.is_safe(self.get_current_waypoint()['joint'], joint_radian):
            return self.robot.move_joint(joint_radian,issync=True)
        else:
            raise ValueError("The motion is not safe! The joint difference exceeds the threshold.")
    
    def move_line(self, joint_radian):
        """
        对傲博move_line函数的封装，添加了安全限位

        Args:
            joint_radian (tuple):list of 6 float, 关节角度，单位为弧度

        Returns:
            ret : the result of move_line operation
        """
        joint_radian = tuple(joint_radian)

        if self.is_safe(self.get_current_waypoint()['joint'], joint_radian):
            return self.robot.move_line(joint_radian)
        else:
            raise ValueError("The motion is not safe! The joint difference exceeds the threshold.")
    
    def movel(self, pos, ori, joint = False):
        '''
        pos: list of 3 float[x,y,z]
        ori: list of 3 float[roll,pitch,yaw]
        joint: bool 是否使用底层关节空间移动
        位姿是机械臂的法兰坐标相对于基座坐标
        '''
        quaternion = rpy2quaternion_standard(np.array(ori)) # rpy-> [qx,qy,qz,qw] -> [qw,qx,qy,qz]

        current_joint_state = self.get_current_waypoint()['joint']
        try:
            result = self.robot.inverse_kin(current_joint_state, pos, quaternion)
            
            if result is not None:
               
                target_joint_radian = result['joint']
                ret = self.movej(target_joint_radian) if joint else self.move_line(target_joint_radian)
                return ret
            else :
                raise ValueError("The inverse kinematics solution failed!")
        
        except ValueError:
            self.move_stop()
    
    def move_stop(self):
        return self.robot.move_stop()
    
    def get_current_waypoint(self):
        """
        * FUNCTION:    get_current_waypoint
        * DESCRIPTION: 获取机械臂当前位置信息
        * INPUTS:      grade碰撞等级:碰撞等级 范围（0～10）
        * OUTPUTS:
        * RETURNS:     成功返回: 关节位置信息，结果为详见NOTES
        *              失败返回: None
        *
        * NOTES:       六个关节角 {'joint': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        *              位置 'pos': [-0.06403157614989634, -0.4185973810159096, 0.816883228463401],
        *              姿态 'ori': [-0.11863209307193756, 0.3820514380931854, 0.0, 0.9164950251579285]}
        *                   (qw, qx, qy, qz)四元数
        """
        # 编码器底层当是做了滤波处理，其波动并不明显，并且在0.001mm级别 
        waypoint = self.robot.get_current_waypoint()
        
        return waypoint

    def set_joint_maxacc(self, joint_maxacc):
        return self.robot.set_joint_maxacc(joint_maxacc)
    def set_joint_maxvelc(self, joint_maxvelc):
        return self.robot.set_joint_maxvelc(joint_maxvelc)
    def set_arrival_ahead_blend(self, blend_radius):
        return self.robot.set_arrival_ahead_blend(blend_radius)
    def set_end_speed(self, end_speed):
        return self.robot.set_end_max_line_velc(end_speed)
    def set_end_acc(self, end_acc):
        return self.robot.set_end_max_line_acc(end_acc)



if __name__== "__main__":
    from ConstConfig import Const
    from Transform import *
    from copy import deepcopy

    ip = Const.Robot.IP
    port = Const.Robot.PORT
    joint_maxacc = Const.Robot.JOINT_MAX_ACC
    joint_maxvelc = Const.Robot.JOINT_MAX_VELC
    end_max_acc = Const.Robot.END_MAX_ACC/2
    end_max_velc = Const.Robot.END_MAX_VELC/2

    hand_in_eye_offset = Const.Robot.HAND_IN_EYE_OFFSET

    aubo = AuboController(ip=ip, port=port)
    # 设置运动参数
    aubo.set_joint_maxacc(joint_maxacc)
    aubo.set_joint_maxvelc(joint_maxvelc)
    aubo.set_end_speed(end_max_velc)
    aubo.set_end_acc(end_max_acc)

    current_waypoint = aubo.get_current_waypoint()
    print(current_waypoint)

    # 开启此处确定标定平面
    input('请确定已经将机器人移动到和齿轮标定板相当高度，按下任意键继续...')
    current_pos = current_waypoint['pos']
    current_rpy = quaternion_standard2rpy(current_waypoint['ori']) 

    init_pos = deepcopy(current_pos)
    init_pos[2] -= hand_in_eye_offset[2]
    init_ori = deepcopy(current_rpy)

    input(f'目标姿态:pos {init_pos}, 姿态: {init_ori}，按下任意键继续...')
    aubo.movel(init_pos, init_ori, joint=False )
    print(f"移动到目标姿态: {init_pos}, 姿态: {init_ori}")
    aubo.disconnect()
