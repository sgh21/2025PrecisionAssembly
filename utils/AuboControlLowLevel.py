from Aubo5i.robotcontrol import *
from Transform import *
import copy

class AuboController:
    def __init__(self, ip='192.168.70.100', port=8899):
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

        # 打印上下文
        logger.info("robot.rshd={0}".format(self.handle))

        self.queue = Queue()

        self.p = Process(target=runWaypoint, args=(self.queue,))
        self.p.start()
        print("Logger process started.")

        # 连接机器人
        self.connect(ip, port)

    def connect(self, ip, port):
        try:
            result =self.robot.connect(ip, port)
            print(result)
            if result != RobotErrorType.RobotError_SUCC:
                logger.info("connect to server{0}:{1} failed.".format(ip, port))
            else:
                logger.info("connected to server{0}:{1} successed.".format(ip, port))
                self.robot.enable_robot_event()
                self.robot.init_profile()
                
                self.robot.set_end_max_line_acc(0.1)
                self.robot.set_end_max_line_velc(0.12)

        except KeyboardInterrupt:
            self.robot.move_stop()

        except RobotError as e:
            logger.error("robot Event:{0}".format(e))

    def disconnect(self):
        if self.robot.connected:
            # 断开机械臂链接
            self.robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        print("run end-------------------------")
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
        print("current_joint ",current * 180/np.pi)
        print('target_joint', target * 180 / np.pi)
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
        quaternion = quaternion2standard(rpy2quaternion(np.array(ori))) # rpy-> [qx,qy,qz,qw] -> [qw,qx,qy,qz]

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
    ip = '192.168.70.100'
    port = 8899
    aubo = AuboController(ip = ip, port = port )
    print("AuboController initialized.")

    # 设置运动参数
    aubo.set_joint_maxacc([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    aubo.set_joint_maxvelc([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    aubo.set_end_speed(0.1)
    aubo.set_end_acc(0.1)

    print(aubo.get_current_waypoint())
    delta_joint = np.array([5, 5, 5, 0, 5, 5]) * np.pi / 180  # 转换为弧度
    current_joint = aubo.get_current_waypoint()['joint']
    target_joint = np.array(current_joint) + delta_joint
    print("Current joint angles:", current_joint)
    print("Target joint angles:", target_joint)

    # 执行关节空间运动
    try:
        aubo.movej(target_joint)
        print("Moved to target joint angles successfully.")
    except ValueError as e:
        print(f"Error during joint movement: {e}")

    # 执行笛卡尔空间运动
    delta_pos = [0.0, 0.0, 0.05]  # 目标位置
    delta_ori = [0.0, 0.0, 10 / 180 *np.pi]  # 目标姿态（欧拉角）
    current_waypoint = aubo.get_current_waypoint()
    current_pos = current_waypoint['pos']
    current_ori = current_waypoint['ori']
    current_ori_qu = standard2quaternion(current_ori)  # 转换为标准四元数形式
    current_ori_rpy = quaternion2rpy(current_ori_qu)  # 转换为欧拉角形式
    target_pos = np.array(current_pos) + np.array(delta_pos)
    target_ori = np.array(current_ori_rpy) + np.array(delta_ori)
    print("Current position:", current_pos)
    print("Target position:", target_pos)
    try:
        aubo.movel(target_pos, target_ori, joint=True)
        # print("Target joint angles:", np.array(result) * 180 / np.pi)  # 转换为角度
        print("Moved to target position successfully.")
    except ValueError as e:
        print(f"Error during Cartesian movement: {e}")
    
    aubo.disconnect()
