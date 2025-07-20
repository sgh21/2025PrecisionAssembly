from ultralytics import YOLO
import torch
import numpy as np
import cv2
import os
import sys

current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))

from src.ComputePose import detect_img
from utils.SegmentResult import Const
from utils.MYtransform import *
from utils.MVSControl import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO("../weights/yolov8-hole.pt")  # Load a trained model

if __name__ == '__main__':

    cam = MVSController()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0])

    while True:

        img = cam.get_image()
        if img is None:
            continue

        gear_pos, hole_pos_list, gear_angle, show_img = detect_img(img, model, show=True)

        cv2.imshow("image", show_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    cam.close_device()

