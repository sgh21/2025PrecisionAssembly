import os, sys
import cv2
import torch
from ultralytics import YOLO

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(workspace)

from configs.ConstConfig import Const
from utils.ComputePoseV4 import ImageProcessor  # 使用V4版本：YOLO-gear, SAM-hole/keyhole

device = "cuda" if torch.cuda.is_available() else "cpu"

# 配置
# img_dir = os.path.join(workspace, './documents/dataset/0913/images')  # 现场照片
# img_dir = os.path.join(workspace, './documents/dataset/1113/images')  # 全部照片
# img_dir = os.path.join(workspace, './documents/dataset/官网')  # 官网照片
img_dir = os.path.join(workspace, './test/images')      # 不旋转末端的照片

YOLO_WEIGHTS = os.path.join(workspace, "models/yolov11s-seg-1113-2.pt")

SAM_MODEL_TYPE = Const.Sam.SAM_MODEL_TYPE
SAM_WEIGHTS = os.path.join(workspace, Const.Sam.MODEL_DIR, Const.Sam.SAM_WEIGHTS)


# 加载模型

img_processor = ImageProcessor(
                device=device, yolo_model_weights=YOLO_WEIGHTS,
                model_weights=SAM_WEIGHTS, model_type=SAM_MODEL_TYPE, show=False, waitkey=20)

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
    gear_pos, hole_pos_list, _, _, result_img = img_processor.process_image(img)
    show_img = cv2.resize(result_img, (result_img.shape[1]//3, result_img.shape[0]//3))
    cv2.imshow('YOLO Inference', show_img)
    print(f"推理图片: {img_files[idx]}，按空格查看下一张，ESC退出。")
    print(f"孔洞相对齿轮位置: {[[h[0]-gear_pos[0], h[1]-gear_pos[1]] for h in hole_pos_list]}")

    key = cv2.waitKey(0)
    if key == 27:  # ESC退出
        break
    elif key == 32:  # 空格下一张
        idx += 1
    else:
        # 其它按键不处理，继续等待
        continue

cv2.destroyAllWindows()