import os, sys
import cv2
from ultralytics import YOLO
workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(workspace)

from configs.ConstConfig import Const
from utils.ComputePoseV3 import ImageProcessor

# 配置

# 配置
# img_dir = os.path.join(workspace, './documents/dataset/0913/images')  # 现场照片
# img_dir = os.path.join(workspace, './documents/dataset/1113/images')  # 全部照片
img_dir = os.path.join(workspace, './documents/dataset/官网')  # 官网照片
model_path = os.path.join(workspace, './models/yolov11s-seg-1113-2.pt')      # 修改为你的模型权重路径
# 加载模型
model = YOLO(model_path)

# 获取所有图片文件
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"找到 {len(img_files)} 张图片用于推理.")

idx = 0
while idx < len(img_files):
    img_path = os.path.join(img_dir, img_files[idx])
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        idx += 1
        continue

    # YOLO推理
    results = model(img)
    result_img = results[0].plot()
    show_img = cv2.resize(result_img, (result_img.shape[1]//3, result_img.shape[0]//3))
    cv2.imshow('YOLO Inference', show_img)
    print(f"推理图片: {img_files[idx]}，按空格查看下一张，ESC退出。")

    key = cv2.waitKey(0)
    if key == 27:  # ESC退出
        break
    elif key == 32:  # 空格下一张
        idx += 1
    else:
        # 其它按键不处理，继续等待
        continue

cv2.destroyAllWindows()