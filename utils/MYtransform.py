import numpy as np


R2D=180/np.pi

def quaternion_to_rpy(quat):# 四元数(w,x,y,z)转为rpy角
    w,x,y,z=quat
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]

def rpy_to_quaternion(rpy):# rpy角(roll, pitch, yaw)转为四元数(w,x,y,z)
    roll,pitch,yaw=rpy
    cr=np.cos(roll/2)
    sr=np.sin(roll/2)
    cp=np.cos(pitch/2)
    sp=np.sin(pitch/2)
    cy=np.cos(yaw/2)
    sy=np.sin(yaw/2)
    w=cy*cp*cr+sy*sp*sr
    x=cy*cp*sr-sy*sp*cr
    y=sy*cp*sr+cy*sp*cr
    z=sy*cp*cr-cy*sp*sr
    return [w,x,y,z]

def quaternion_to_standard(quat):# 四元数(w,x,y,z)转为标准四元数(x,y,z,w)
    return [quat[1],quat[2],quat[3],quat[0]]

def standard_to_quaternion(quat):# 标准四元数(x,y,z,w)转为四元数(w,x,y,z)
    return [quat[3],quat[0],quat[1],quat[2]]
