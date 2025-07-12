import os
import sys
import cv2
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(current_dir)
sys.path.append(workspace)

from utils.MYtransform import *
from utils.MVSControl import *
# from utils.MYAuboControl import AuboController


if __name__ == '__main__':
    cam = MVSController()
    
    img = cam.capture_frame()

    # cv2.imshow('img', img)
    cv2.waitKey(0)
    
    # aubo=AuboController()
    # aubo.connect('192.168.70.100',8899)
    # aubo.robot.set_end_max_line_acc(0.05)
    # aubo.robot.set_end_max_line_velc(0.05)
    # initpoint = aubo.robot.get_current_waypoint()
    # init_pos = initpoint['pos']
    # init_ori = initpoint['ori']
    # init_ori = quaternion_to_rpy(np.array(init_ori))
    # print(init_pos,init_ori)

    # capture_points = []

    # for i in range(1,11):
    #     ori = i*10/R2D
    #     capture_points.append({'pos': init_pos, 'ori': [x + y for x, y in zip(init_ori, [0, 0, ori])]})

    # # for j in range(4):
    # for i in range(1, 1+len(capture_points)):
    #     aubo.move_line(capture_points[i]['pos'], capture_points[i]['ori'])
    #     time.sleep(1)
    #     img = cam.get_image()
    #     img=cv2.resize(img,(1280,960))
    #     cv2.imwrite(f'{workspace}/training/dataset/originimg/gear/img_{i}.jpg',img)
    #     print(f'img_{i} finished')
    #     time.sleep(1)
    # aubo.move_line(capture_points[i]['pos'], capture_points[i]['ori'])
    #     # time.sleep(10)


    # aubo.move_line(init_pos,init_ori)

    cam.close_device()