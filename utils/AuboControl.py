from Aubo5i.robotcontrol import *
from Transform import *
from TF import *
from utils.config import CAMERA_CONFIG

workspace = os.path.dirname(os.path.abspath(__file__))

class AuboController:
    def __init__(self):
        self.tf_tree = TFTree()
        self.tf_init = False
        self.camera2flange_init = False
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

    def connect(self,ip,port):
        try:
            # 链接服务器
            #ip = 'localhost'
            ip = ip
            port = port
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

                # 傻鸟傲博，下边这俩函数没用
                # joint_maxvelc = (2.596177, 2.596177, 2.596177, 3.110177, 3.110177, 3.110177)
                # joint_maxacc = (17.308779/1000, 17.308779/1000, 17.308779/1000, 17.308779/1000, 17.308779/1000, 17.308779/1000)
                # self.robot.set_joint_maxacc(joint_maxacc)
                # self.robot.set_joint_maxvelc(joint_maxvelc)

        except KeyboardInterrupt:
            self.robot.move_stop()

        except RobotError as e:
            logger.error("robot Event:{0}".format(e))

    def get_camera2flange(self,config_path=CAMERA_CONFIG):
        import yaml 
        file_path = os.path.join(workspace,config_path)
        with open(file_path,"r") as f:
            camera_config = yaml.safe_load(f)
            intrinsics = np.array(camera_config["intrinsic"])
            distortion = np.array(camera_config["distortion_coefficients"])
            camera2flange = np.array(camera_config["extrinsic"])
            self.camera2flange_init = True
        
        return intrinsics,distortion,camera2flange
    def get_gripper1to2(self,config_path=CAMERA_CONFIG):
        import yaml 
        file_path = os.path.join(workspace,config_path)
        with open(file_path,"r") as f:
            camera_config = yaml.safe_load(f)
            if "T_gripper1to2" not in camera_config:
                return None
            T_gripper1to2 = np.array(camera_config["T_gripper1to2"])
            
            self.camera2flange_init = True
        return T_gripper1to2
    def init_tf_tree(self):
        tf_params = {
            "camera2flange": {
                "child_name":"camera_center",
                "parent_name":"flange_center",
                "translation": [0.000937, 0.109529, 0.0118177],
                "rotation": [-0.00565719,-0.00407096,0.9999521,0.00687191]
            },
            "gripper2flange": {
                "child_name":"gripper_center",
                "parent_name":"flange_center",
                "translation": [0,0,0.167],
                "rotation": [0,0,0,1]
            },
            "target2world": {
                "child_name":"target",
                "parent_name":"world",
                "translation": [0,0,0],
                "rotation": [0,0,0,1]
            },
            "cube2camera": {
                "child_name":"cube_refer",
                "parent_name":"camera_center",
                "translation": [0,0,0],
                "rotation": [0,0,1,0]
            },
            "gripper1to2":{
                "child_name":"gripper_affine",
                "parent_name":"gripper_center",
                "translation":[0,0,0],
                "rotation":[0,0,0,1]
            }
        }
        if not self.camera2flange_init:
            print("Init the camera2flange")
            intrinsics,distortion,camera2flange = self.get_camera2flange()
            T_gripper1to2 = self.get_gripper1to2()
            tf_params["camera2flange"]["translation"] = camera2flange[:3,3]
            tf_params["camera2flange"]["rotation"] = R.from_matrix(camera2flange[:3,:3]).as_quat() 
            if T_gripper1to2 is not None:
                tf_params["gripper1to2"]["translation"] = T_gripper1to2[:3,3]
                tf_params["gripper1to2"]["rotation"] = R.from_matrix(T_gripper1to2[:3,:3]).as_quat()
        print(tf_params['camera2flange'])
        if self.robot.connected:
            waypoint = self.get_current_waypoint()
            pos = waypoint['pos']
            ori = waypoint['ori']
            # print("The current pose is :",pos,ori)
            try:
                self.tf_tree.add_node("flange_center","world",pos,ori)
                for joint in tf_params:
                   
                    self.tf_tree.add_node(**tf_params[joint])
                    
                self.tf_init = True
            except ValueError:
                raise ValueError("Cannot add node to tf tree.")
        else:
            raise ValueError("The robot is not connected.")
        
        self.tf_tree.print_tree_structure()
        return self.tf_tree
    
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
        SB函数没球用
        """
        return self.robot.set_tool_kinematics_param(tool_kinematics_param)
    
    def movej(self,joint_radian):
        joint_radian = tuple(joint_radian)
        return self.robot.move_joint(joint_radian,issync=True)
    def move_stop(self):
        return self.robot.move_stop()
    
    def get_current_waypoint(self):
        '''
        return waypoint and convert ori to standard
        '''
        waypoint = self.robot.get_current_waypoint()
        ori = waypoint['ori']
        waypoint['ori'] = quaternion_to_standard(np.array(ori))
        
        return waypoint
    
    def movel(self,pos,ori):
        '''
        pos: list of 3 float[x,y,z]
        ori: list of 3 float[roll,pitch,yaw]
        '''
        quaternion = rpy_to_quaternion(np.array(ori)) #[x,y,z,w]
        quaternion = standard_to_quaternion(quaternion)#[w,x,y,z]   
        current_joint_state = self.get_current_waypoint()['joint']
        try:
            result = self.robot.inverse_kin(current_joint_state, pos, quaternion)
            if result is not None:
                joint_radian = result['joint']
                result = self.robot.move_line(joint_radian)
                self.update_flange_center()
                # print("The pos will be moved to :",self.robot.forward_kin(joint_radian))
                return result
            else:
                raise ValueError("inverse kinematics failed.")
        except ValueError:
            self.robot.move_stop()

    def update_flange_center(self):
        waypoint = self.get_current_waypoint()
        flange_pos = waypoint['pos']
        flange_ori = waypoint['ori']
        
        self.tf_tree.update_node("flange_center",flange_pos,flange_ori)

    def get_pose(self,frame_name,refer_frame="world"):
        '''
        get the pose of the frame_name in the reference frame
        frame_name: str
        return: pos[x,y,z]
                ori[x,y,z,w]
        '''
        if not self.tf_init:
            self.init_tf_tree()
        if frame_name not in self.tf_tree.nodes:
            raise ValueError(f"Frame node {frame_name} not found.")
        # 更新flange_center节点的位姿，保证正确转换
        self.update_flange_center()
        T = self.tf_tree.get_transform(frame_name,refer_frame)
        # T = self.tf_tree._get_transform_to_world(frame_name)
        pos,ori = self.tf_tree.transform_to_pose(T)
        # print(f"The pose of {frame_name} in {refer_frame} is :",pos,ori)
        return pos,ori 
    
    def movel_tf(self,pos,ori,frame_name="flange_center",joint=False):
        ''' 
        move the frame to target pose in the world frame
        pos: list of 3 float[x,y,z]
        ori: list of 3 float[roll,pitch,yaw]
        '''
        ori = rpy_to_quaternion(np.array(ori)) #[x,y,z,w]
        if not self.tf_init:
            self.init_tf_tree()
        # 更新flange_center节点的位姿，保证正确转换
        self.update_flange_center()# 建议用到坐标转换时都更新一下flange_center的位姿
        if frame_name not in self.tf_tree.nodes:
            raise ValueError(f"Frame node {frame_name} not found.")
        # 更新target节点的位姿
        self.tf_tree.update_node("target",pos,ori)
        # 将frame_name和target的位姿重合，解算flange_center对世界的位姿
        T_flange2frame = self.tf_tree.get_transform("flange_center",frame_name)
        T_target2world = self.tf_tree.get_transform("target","world")
        T = T_target2world @ T_flange2frame
        pos,ori = self.tf_tree.transform_to_pose(T)#[x,y,z],[x,y,z,w]
        
        current_joint_state = self.get_current_waypoint()['joint']
        try:
            self.tf_tree.update_node("flange_center",pos,ori)#[x,y,z,w]
            ori = standard_to_quaternion(ori) #[w,x,y,z]
            
            result = self.robot.inverse_kin(current_joint_state, pos,ori)
            
            if result is not None :
                joint_radian = result['joint']
                # print("I run here")
            #     joint = np.array(result['joint'])*180/np.pi
            # # logger.info("inverse kinematics result is :{0}".format(joint))
            #     print('The inverse kinematics result is:',joint)
                if joint == False:
                    return self.robot.move_line(joint_radian)
                else:
                    return self.robot.move_joint(joint_radian)
               
            else:
                raise ValueError("inverse kinematics failed.")
        except ValueError:
            self.robot.move_stop()

    def movel_relative(self,pos,ori,frame_name="flange_center",joint=False):
        '''
        move to the frame to target pose in the self frame
        pos: list of 3 float[x,y,z]
        ori: list of 3 float[roll,pitch,yaw]
        '''
    
        ori = rpy_to_quaternion(np.array(ori)) #[x,y,z,w]
        # 更新flange_center节点的位姿，保证正确转换
        self.update_flange_center()# 建议用到坐标转换时都更新一下flange_center的位姿
        self.tf_tree.add_node("temp",frame_name,pos,ori)
        T = self.tf_tree.get_transform("temp","world")
        target_pos,target_ori = self.tf_tree.transform_to_pose(T)#[x,y,z],[x,y,z,w]
        target_ori = quaternion_to_rpy(np.array(target_ori))
        self.movel_tf(target_pos,target_ori,frame_name,joint=joint)
        self.tf_tree.delete_node("temp")

    def move_trajectory(self,waypoints,frame_name = 'flange_center',type=RobotMoveTrackType.CARTESIAN_MOVEP):
        '''
        waypoints: list of dict [{'pos':[x,y,z],'ori':[roll,pitch,yaw]},...]
        type: RobotMoveTrackType
        '''
        if not self.tf_init:
            self.init_tf_tree()
        self.update_flange_center()
        if frame_name not in self.tf_tree.nodes:
            raise ValueError(f"Frame node {frame_name} not found.")
        
        for waypoint in waypoints:
            pos = waypoint['pos']
            ori = waypoint['ori']
            ori = rpy_to_quaternion(np.array(ori)) #[x,y,z,w]
            # 更新target节点的位姿
            self.tf_tree.update_node("target",pos,ori)
            # 将frame_name和target的位姿重合，解算flange_center对世界的位姿
            T_flange2frame = self.tf_tree.get_transform("flange_center",frame_name)
            T_target2world = self.tf_tree.get_transform("target","world")
            T = T_target2world @ T_flange2frame
            pos,ori = self.tf_tree.transform_to_pose(T)#[x,y,z],[x,y,z,w]
            
            current_joint_state = self.get_current_waypoint()['joint']

            try:
                self.tf_tree.update_node("flange_center",pos,ori)#[x,y,z,w]
                ori = standard_to_quaternion(ori) #[w,x,y,z]
                joint_radian = self.robot.inverse_kin(current_joint_state, pos,ori)['joint']
                if joint_radian is not None:
                    self.robot.add_waypoint(joint_radian)
                else:
                    raise ValueError("inverse kinematics failed.")
            except ValueError:
                self.robot.move_stop()
        
        self.robot.move_track(track=type)
    
        return self.robot.remove_all_waypoint()
    def set_joint_maxacc(self,joint_maxacc):
        return self.robot.set_joint_maxacc(joint_maxacc)
    def set_joint_maxvelc(self,joint_maxvelc):
        return self.robot.set_joint_maxvelc(joint_maxvelc)
    def set_arrival_ahead_blend(self,blend_radius):
        return self.robot.set_arrival_ahead_blend(blend_radius)
    def set_end_speed(self,end_speed):
        return self.robot.set_end_max_line_velc(end_speed)
    def set_end_acc(self,end_acc):
        return self.robot.set_end_max_line_acc(end_acc)



if __name__== "__main__":
    from utils.config import INITPOS,INITORI,AUBOHOST
    ip = AUBOHOST
    port = 8899
    aubo = AuboController()
    aubo.connect(ip,port)
    # current_joint = np.array([172.89572035,  -5.42523249 ,-99.01730021 ,  1.68216279, -83.85767486,89.78194189])/180*np.pi
    # delta_joint = np.array([10,0,0,0,0,0])/180*np.pi
    # aubo.movej(current_joint+delta_joint)
#     pose = np.array([1.3157066,-5.74676066 ,-90.59072196 ,  5.1560387,  -90.
#    ,1.3157066])/180*np.pi
#     aubo.movej(pose)
    aubo.movel_tf(pos=INITPOS,ori=INITORI,frame_name='flange_center')
    # aubo.movel_relative(pos=np.array([0,0,-0.05]),ori=np.array([0,0,0]),frame_name='flange_center')
    # aubo.movel_tf(pos=INITPOS,ori=INITORI,frame_name='flange_center')
    
    # pos,ori =aubo.get_pose("camera_center")
    # ori = quaternion_to_rpy(np.array(ori))
    # delta_pos = [0.0,0.0,0.0]
    # deltal_ori = np.array([0.0,0,1*np.pi/180])
    # ori_or = [0.0,0.0,0.0]
    
    # for i in range(45):
            
    #         aubo.movel_relative(delta_pos,deltal_ori,frame_name="camera_center")
    #         # # pos = list(pos[i]+flag*delta_pos[i] for i in range(len(delta_pos)))
    #         # target_ori = ori.copy()
    #         # target_ori = list(target_ori[i]+deltal_ori[i] for i in range(len(deltal_ori)))
    #         # aubo.movel_tf(pos,target_ori,frame_name="camera_center")
    #         time.sleep(0.5)
    #         # # aubo.movel_relative(delta_pos,ori_or,frame_name="camera_center")
    #         # # time.sleep(2)
    #         # aubo.movel_tf(pos,ori,frame_name="camera_center")
    #         # time.sleep(2)
    # # else:
    # #     raise ValueError("joint_radian is None, cannot proceed.")
    #     # raise ValueError("joint_radian is None, cannot proceed.")
    
    aubo.disconnect()
