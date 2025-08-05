# author: Shen Guanghui
# date: 2025-06-16
# description: 基于scipy.spatial.transform的角度变换工具，
#              包含欧拉角、四元数、旋转矩阵和旋转向量之间的转换
import numpy as np
from scipy.spatial.transform import Rotation as R

def standard2quaternion(quaternion):
    """
    将四元数从 (qw, qx, qy, qz) 转换为 (qx, qy, qz, qw)标准形式。

    参数:
    傲博标准四元数 -- 四元数 (qw, qx, qy, qz)

    返回:
    quaternion -- (qx, qy, qz, qw)
    """
    if len(quaternion) != 4:
        raise ValueError("quaternion must have 4 elements.")
    return np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])


def quaternion2standard(quaternion):
    """
    将四元数从 (qx, qy, qz, qw) 转换为 (qw, qx, qy, qz)标准形式。

    参数:
    quaternion -- 四元数 (qx, qy, qz, qw)

    返回:
    傲博标准四元数 (qw, qx, qy, qz)
    """
    if len(quaternion) != 4:
        raise ValueError("quaternion must have 4 elements.")
    return np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])


def rpy2quaternion(rpy):
    """
    将 Roll-Pitch-Yaw (RPY) 角转换为四元数。

    参数:
    rpy -- 包含 roll, pitch, yaw 的数组（弧度）

    返回:
    四元数 (qx, qy, qz, qw)
    """
    if len(rpy) != 3:
        raise ValueError("RPY angles must have 3 elements.")
    # xyz的欧拉角对应于roll pitch yaw
    # 这个欧拉角默认是外旋的
    r = R.from_euler('xyz', rpy)
    quaternion = r.as_quat()
    
    return quaternion


def quaternion2rpy(quaternion):
    """
    将四元数转换为 Roll-Pitch-Yaw (RPY) 角。

    参数:
    quaternion -- 四元数 (qx, qy, qz, qw)

    返回:
    RPY 角 (roll, pitch, yaw)
    """
    if len(quaternion) != 4:
        raise ValueError("quaternion must have 4 elements.")
    
    r = R.from_quat(quaternion)
    rpy = r.as_euler('xyz', degrees=False)
    
    return rpy

def rpy2quaternion_standard(rpy):
    """
    将 Roll-Pitch-Yaw (RPY) 角转换为四元数，并转换为傲博标准形式。

    参数:
    rpy -- 包含 roll, pitch, yaw 的数组（弧度）

    返回:
    四元数 (qw, qx, qy, qz,)
    """
    quaternion = rpy2quaternion(rpy)
    return quaternion2standard(quaternion)

def quaternion_standard2rpy(quaternion):
    """
    将傲博标准形式的四元数转换为 Roll-Pitch-Yaw (RPY) 角。

    参数:
    quaternion_standard -- 四元数 (qw, qx, qy, qz)

    返回:
    RPY 角 (roll, pitch, yaw)
    """
    quaternion = standard2quaternion(quaternion)
    return quaternion2rpy(quaternion)

def rpy2rotation(rpy):
    """
    将 Roll-Pitch-Yaw (RPY) 角转换为旋转矩阵。

    参数:
    rpy -- 包含 roll, pitch, yaw 的数组（弧度）

    返回:
    旋转矩阵
    """
    r = R.from_euler('xyz', rpy)
    return r.as_matrix()


def rotation2rpy(rotation):
    """
    将旋转矩阵转换为 Roll-Pitch-Yaw (RPY) 角。

    参数:
    rotation -- 旋转矩阵

    返回:
    RPY 角 (roll, pitch, yaw)（弧度）
    """
    r = R.from_matrix(rotation)
    return r.as_euler('xyz')


def rotation2quaternion(rotation):
    """
    将旋转矩阵转换为四元数。

    参数:
    rotation -- 旋转矩阵

    返回:
    四元数 (qx, qy, qz, qw)
    """
    r = R.from_matrix(rotation)
    return r.as_quat()


def quaternion2rotation(quaternion):
    """
    将四元数转换为旋转矩阵。

    参数:
    quaternion -- 四元数 (qx, qy, qz, qw)

    返回:
    旋转矩阵
    """
    r = R.from_quat(quaternion)
    return r.as_matrix()


def rotation2axis_angle(rotation):
    """
    将旋转矩阵转换为轴角表示。

    参数:
    rotation -- 旋转矩阵

    返回:
    轴角表示
    """
    r = R.from_matrix(rotation)
    return r.as_rotvec()


def axis_angle2rotation(axis_angle):
    """
    将轴角表示转换为旋转矩阵。

    参数:
    axis_angle -- 轴角表示

    返回:
    旋转矩阵
    """
    r = R.from_rotvec(axis_angle)
    return r.as_matrix()


def quaternion2axis_angle(quaternion):
    """
    将四元数转换为轴角表示。

    参数:
    quaternion -- 四元数 (qx, qy, qz, qw)

    返回:
    轴角表示
    """
    r = R.from_quat(quaternion)
    return r.as_rotvec()


def axis_angle2quaternion(axis_angle):
    """
    将轴角表示转换为四元数。

    参数:
    axis_angle -- 轴角表示

    返回:
    四元数 (qx, qy, qz, qw)
    """
    r = R.from_rotvec(axis_angle)
    return r.as_quat()


def rpy2axis_angle(rpy):
    """
    将 Roll-Pitch-Yaw (RPY) 角转换为轴角表示。

    参数:
    rpy -- 包含 roll, pitch, yaw 的数组（弧度）

    返回:
    轴角表示
    """
    r = R.from_euler('xyz', rpy)
    return r.as_rotvec()


def axis_angle2rpy(axis_angle):
    """
    将轴角表示转换为 Roll-Pitch-Yaw (RPY) 角。

    参数:
    axis_angle -- 轴角表示

    返回:
    RPY 角 (roll, pitch, yaw)（弧度）
    """
    r = R.from_rotvec(axis_angle)
    return r.as_euler('xyz')

def gear_angle2controller_angle(gear_angle, center_connection_angle, tooth_range = 20 / 180 * np.pi, normal_to_zero = True):
    """
    根据一个齿轮角度计算另一个齿轮角度，默认在连线角度为0的情况下，两个齿轮相对角度为0。

    参数:
    gear_angle -- 齿轮角度（弧度）
    center_connection_angle -- 中心连线角度（弧度）
    tooth_range -- 齿间角度（弧度）
    normal_to_zero -- 是否将角度归一化到(-tooth_range/2, tooth_range/2]范围内
    返回:
    controller_angle -- 另一齿轮角度（弧度）
    """

    controller_angle = ( -gear_angle + 2 * center_connection_angle ) % (tooth_range)

    if normal_to_zero and controller_angle > tooth_range / 2:
            controller_angle -= tooth_range
    
    return controller_angle
if __name__ == "__main__":
    # 示例调用
    roll = np.radians(30)  # 30 度转换为弧度
    pitch = np.radians(45)  # 45 度转换为弧度
    yaw = np.radians(75)  # 75 度转换为弧度
    
    # 测试四元数转换
    qx, qy, qz, qw = rpy2quaternion(np.array([roll, pitch, yaw]))
    print(f"Quaternion: ({qx}, {qy}, {qz}, {qw})")
    
    # 测试RPY转换
    r, p, y = quaternion2rpy(np.array([qx, qy, qz, qw]))
    print(f"RPY: ({np.degrees(r)}, {np.degrees(p)}, {np.degrees(y)})")