import sys, os
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import datetime
import cv2

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workspace)

from configs.ConstConfig import Const
from utils.AuboControlLowLevel import AuboController
from utils.MVSControl import MVSController

CALIB_POS = Const.Robot.CALIB_POS
GEAR_POS = Const.Robot.INIT_POS
INIT_ORI = Const.Robot.INIT_ORI
HAND_IN_EYE_OFFSET = Const.Robot.HAND_IN_EYE_OFFSET
GEAR_POS = [GEAR_POS[0]+HAND_IN_EYE_OFFSET[0], GEAR_POS[1]+HAND_IN_EYE_OFFSET[1], GEAR_POS[2]]
CALIB_POS = [CALIB_POS[0]+HAND_IN_EYE_OFFSET[0], CALIB_POS[1]+HAND_IN_EYE_OFFSET[1], CALIB_POS[2]]

RADIUS = 0.08
OFFSET = 0.01

positions = [CALIB_POS,
             GEAR_POS]

for i in range(6):
    theta = np.pi/3 * i
    x = GEAR_POS[0] + RADIUS * np.cos(theta)
    y = GEAR_POS[1] + RADIUS * np.sin(theta)
    z = GEAR_POS[2]
    positions.append([x, y, z])

print("采集位置：", positions)

if __name__ == "__main__":
    ip = Const.Robot.IP
    port = Const.Robot.PORT
    aubo = AuboController(ip=ip, port=port)
    mvs = MVSController()
    img_dir = './capture_data/'
    os.makedirs(img_dir, exist_ok=True)

    # 每个对象附近分别采样
    for pos in positions:
        print(f"在目标位置采样: {pos}")
        for i in range(5):
            camera_pos = pos.copy()
            target_ori = INIT_ORI.copy()
            camera_pos[0] += np.random.randn() * OFFSET
            camera_pos[1] += np.random.randn() * OFFSET

            # target_ori[2] = np.random.uniform(-np.pi, 0)
            ori_offset = np.random.uniform(-np.pi/2, np.pi/2)
            target_ori[2] += ori_offset

            # 解算机械臂末端位置
            rot = R.from_euler('xyz', [0, 0, ori_offset])
            offset_in_base = rot.apply(HAND_IN_EYE_OFFSET)
            end_pos = (np.array(camera_pos) - offset_in_base).tolist()  # 末端位置

            # end_pos = (np.array(camera_pos) - np.array(HAND_IN_EYE_OFFSET)).tolist()  # 末端位置
            end_pos[2] = GEAR_POS[2]    # 保持Z轴高度不变

            aubo.movel(end_pos, target_ori, joint=True)
            print(f"Capture image at POS:{end_pos}, ORI:{target_ori}")
            time.sleep(1)
            img = mvs.get_image()
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{img_dir}{now_str}.jpg", img)
            
            img = cv2.resize(img, (640, 480))
            cv2.imshow("capture", img)
            key = cv2.waitKey(50)
            time.sleep(1)
    cv2.destroyAllWindows()

