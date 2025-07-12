import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(current_dir)
sys.path.append(os.path.join(workspace))

from lib.Aubo5i.robotcontrol import *
from MYtransform import *
from MYTF import *
from config import *

class AuboController:
    def __init__(self):
        self.tf_tree = TFTree()
        self.tf_init = False

        Auboi5Robot.initialize()    # 系统初始化
        self.robot = Auboi5Robot()   # 创建机械臂控制类

        logger_init()
        self.handle = self.robot.create_context()# 创建上下文
        logger.info("robot.rshd={0}".format(self.handle))# 打印上下文

        print("Logger process started.")

    def connect(self,ip='192.168.70.100',port=8899):
        ''''
        根据ip和端口连接aubo机械臂
        '''
        result=self.robot.connect(ip,port)  # 0:连接成功，无错误
        print(f"connect result={result}")
        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect to server{0}:{1} failed.".format(ip, port))
            raise ValueError("connect to server{0}:{1} failed.".format(ip, port))
        else:
            logger.info("connected to server{0}:{1} successed.".format(ip, port))
            self.robot.enable_robot_event()
            self.robot.init_profile()
            # 设置最大末端速度和加速度
            self.robot.set_end_max_line_acc(0.01)
            self.robot.set_end_max_line_velc(0.012)
    
    def disconnect(self):
        if self.robot.connected:
            # 断开机械臂链接
            self.robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        print("---------------------run end-------------------------")
        self.queue.put('quit')

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
        '''
        初始化TF树，设置节点之间的关系，创建world的孩子flange
        '''
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
        # if not self.camera2flange_init:
        #     print("Init the camera2flange")
        #     intrinsics,distortion,camera2flange = self.get_camera2flange()
        #     T_gripper1to2 = self.get_gripper1to2()
        #     tf_params["camera2flange"]["translation"] = camera2flange[:3,3]
        #     tf_params["camera2flange"]["rotation"] = R.from_matrix(camera2flange[:3,:3]).as_quat() 
        #     if T_gripper1to2 is not None:
        #         tf_params["gripper1to2"]["translation"] = T_gripper1to2[:3,3]
        #         tf_params["gripper1to2"]["rotation"] = R.from_matrix(T_gripper1to2[:3,:3]).as_quat()
        # print(tf_params['camera2flange'])
        if self.robot.connected:
            waypoint = self.robot.get_current_waypoint()
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


    def move_joint(self,destjoint):
        self.robot.move_joint(destjoint)

    def move_line(self,pos,ori):
        current_joint= self.robot.get_current_waypoint()['joint']
        ori=rpy_to_quaternion(ori)  # [w,x,y,z]
        # ori=quaternion_to_standard(ori)
        joint_radian=self.robot.inverse_kin(current_joint,pos,ori)['joint']
        if(joint_radian is None):
            raise ValueError("inverse kinematics failed.")
        self.robot.move_line(joint_radian)

    def move_trajectory(self,waypoints,frame_name = 'world',type=RobotMoveTrackType.CARTESIAN_MOVEP):
        '''
        waypoints: list of dict [{'pos':[x,y,z],'ori':[roll,pitch,yaw]},...]
        type: RobotMoveTrackType
        '''
        # if not self.tf_init:
        #     self.init_tf_tree()
        # self.update_flange_center()
        # if frame_name not in self.tf_tree.nodes:
        #     raise ValueError(f"Frame node {frame_name} not found.")
        for waypoint in waypoints:
            pos = waypoint['pos']
            ori = waypoint['ori']
            ori = rpy_to_quaternion(ori) #[w,x,y,z]
            current_joint = self.robot.get_current_waypoint()['joint']
            try:
                joint_radian = self.robot.inverse_kin(current_joint, pos,ori)['joint']
                if joint_radian is not None:
                    self.robot.add_waypoint(joint_radian)
                else:
                    raise ValueError("inverse kinematics failed.")
            except ValueError:
                self.robot.move_stop()
        self.robot.move_track(track=type)

    def move_target(self,pos,ori,frame_name="flange_center",joint=False):
        '''     # 移动末端(flange_center)至目标位姿(world坐标系)
        move the frame to target pose in the world frame
        pos: list of 3 float[x,y,z]
        ori: list of 3 float[roll,pitch,yaw]
        '''
        if not self.tf_init:
            self.init_tf_tree()
        if frame_name not in self.tf_tree.nodes:
            raise ValueError(f"Frame node {frame_name} not found.")
        ori = rpy_to_quaternion(ori) #[w,x,y,z]
        ori = quaternion_to_standard(ori) #[x,y,z,w]
        self.update_flange_center()# 建议用到坐标转换时都更新一下flange_center的位姿
        # 更新target节点的位姿
        self.tf_tree.update_node("target",pos,ori)
        # 将frame_name和target的位姿重合，解算flange_center对世界的位姿
        T_flange2frame = self.tf_tree.get_transform("flange_center",frame_name)
        T_target2world = self.tf_tree.get_transform("target","world")
        T = T_target2world @ T_flange2frame         # T_flange2frame的作用是如果target是gripper等的目标点，也可以实现定位
        pos,ori = self.tf_tree.transform_to_pose(T)#[x,y,z],[x,y,z,w]
        
        current_joint = self.robot.get_current_waypoint()['joint']
        try:
            self.tf_tree.update_node("flange_center",pos, ori)#[x,y,z,w]
            ori = standard_to_quaternion(ori) #[w,x,y,z]
            result = self.robot.inverse_kin(current_joint, pos, ori)
            if result is not None :
                joint_radian = result['joint']
                if joint == False: 
                    return self.robot.move_line(joint_radian)
                else:
                    return self.robot.move_joint(joint_radian)
               
            else:
                raise ValueError("inverse kinematics failed.")
        except ValueError:
            self.robot.move_stop()

    def move_relative_target(self,rel_pos,rel_ori,reference_frame="flange_center",joint=False):
        '''
        move the frame to the relative pose in the reference frame
        rel_pos: list of 3 float[x,y,z]
        rel_ori: list of 3 float[roll,pitch,yaw]
        '''
        if not self.tf_init:
            self.init_tf_tree()
        if reference_frame not in self.tf_tree.nodes:
            raise ValueError(f"Frame node {reference_frame} not found.")

        self.update_flange_center()
        rel_ori = rpy_to_quaternion(rel_ori)    #[w,x,y,z]
        rel_ori = quaternion_to_standard(rel_ori)   #[x,y,z,w]
        # 将相对位姿转换为绝对位姿        
        pos,ori=self.tf_tree.transform_pose(rel_pos,rel_ori,reference_frame,"world")    #[x,y,z], [x,y,z,w]
        ori = standard_to_quaternion(ori) #[w,x,y,z]
        ori = quaternion_to_rpy(ori)

        self.move_target(pos,ori,reference_frame,joint)

    def update_flange_center(self):
        waypoint = self.robot.get_current_waypoint()
        flange_pos = waypoint['pos']
        flange_ori = waypoint['ori']
        flange_ori = quaternion_to_standard(flange_ori) # [x,y,z,w] 重要！！
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
        self.update_flange_center() # 更新flange_center节点的位姿，保证正确转换
        T = self.tf_tree.get_transform(frame_name,refer_frame)
        pos,ori = self.tf_tree.transform_to_pose(T) # 将变换矩阵转换为位姿
        return pos,ori 


if __name__ == '__main__':
    controller = AuboController()

    ip="192.168.70.100"
    port=8899
 
    controller.connect(ip,port)

    # cpos,cori=controller.robot.get_current_waypoint()['pos'],controller.robot.get_current_waypoint()['ori']
    # print(f"current pose:{cori}")
    # cori=quaternion_to_rpy(cori)
    # print(f"current pose:{cpos},{[x/np.pi*180 for x in cori]}")

    # dests=[{"pos":[-0.5,0.1,0.4],"ori":[-180/R2D,20/R2D,-80/R2D]},]
    # controller.move_target(dest[0]['pos'],dest[0]['ori'])

    point = controller.robot.get_current_waypoint()
    print(f"current pose:{point['pos']},{point['ori']}")
    
    # relative_dests=[{"pos":[-0.1,0.1,0],"ori":[-10/R2D,0,0]}]
    # controller.move_relative_target(relative_dests[0]['pos'],relative_dests[0]['ori'])
    
    controller.disconnect()




    