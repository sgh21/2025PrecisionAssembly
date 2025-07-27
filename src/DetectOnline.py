from ultralytics import YOLO
from datetime import datetime
import torch
import numpy as np
import cv2
import os
import sys
import logging

current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))

from src.ComputePose import detect_img
from utils.SegmentResult import Const
from utils.MYtransform import *
from utils.MVSControl import *
from utils.MYAuboControl import *

class queue:
    def __init__(self, size=10, arr=[]):
        self.size = size
        self.pointer = 0
        self.data = np.zeros(size)
        s = min(size, len(arr))
        self.data[:s] = arr[:s]

    def add(self, item):
        self.data[self.pointer] = item
        self.pointer = (self.pointer + 1) % self.size

    def get(self):
        return np.mean(self.data)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO("../weights/yolov8-hole.pt")  # Load a trained model

# 配置日志
log_dir = os.path.join(current_path, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建日志文件名（包含时间戳）
log_filename = f"detect_online_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_dir, log_filename)

# 配置日志格式和输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 获取logger
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    cam = MVSController()
    controller = AuboController()

    ip="192.168.70.100"
    port=8899
    controller.connect(ip,port)
    ini_point = controller.robot.get_current_waypoint()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0])

    gear_angle_queue = queue(size=10)
    robot_angle_queue = queue(size=10)

    while True:

        img = cam.get_image()
        if img is None:
            continue

        gear_pos, hole_pos_list, gear_angle, show_img = detect_img(img, model, show=True)
        gear_angle_queue.add(gear_angle)

        point = controller.robot.get_current_waypoint()
        robot_angle = quaternion_to_rpy(np.array(point['ori']))[2]
        robot_angle_queue.add(robot_angle)

        print(f"Gear Angle: {gear_angle*R2D:.2f}, Robot Angle: {np.array(point['ori'])*R2D}")

        cv2.imshow("image", show_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info(f"Gear Angle: {gear_angle_queue.get()*R2D:.2f}, Robot Angle: {robot_angle_queue.get()*R2D:.2f}")
            logger.info("Exiting...")
            cv2.destroyAllWindows()
            break
        elif key == ord('c'):
            logger.info("Starting rotation clockwise...")
            logger.info(f"Gear Angle: {gear_angle_queue.get()*R2D:.2f}, Robot Angle: {robot_angle_queue.get()*R2D:.2f}")
            controller.move_relative_target([0, 0, 0], [0, 0, -10/R2D])
        
        elif key == ord('x'):
            logger.info("Starting rotation anti-clockwise...")
            logger.info(f"Gear Angle: {gear_angle_queue.get()*R2D:.2f}, Robot Angle: {robot_angle_queue.get()*R2D:.2f}")
            controller.move_relative_target([0, 0, 0], [0, 0, 10/R2D])



    controller.disconnect()
    cam.close_device()

