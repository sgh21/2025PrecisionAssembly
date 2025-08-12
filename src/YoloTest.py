import cv2
import os
from ultralytics import YOLO
from ConstConfig import Const

IMG_SHAPE_SHOW = Const.Camera.IMG_SHAPE_SHOW
YOLO_WEIGHTS = os.path.join(Const.Yolo.MODEL_DIE, Const.Yolo.YOLO_HOLE_WEIGHTS)
DATASET_DIR = Const.Data.DATASET_DIR
IMG_DIR = os.path.join(DATASET_DIR, 'yolo_0803/train/images')

def infer_and_show(model, img, win_name="YOLO"):
    results = model(img)
    res_plotted = results[0].plot()
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, IMG_SHAPE_SHOW[1], IMG_SHAPE_SHOW[0])
    cv2.imshow(win_name, res_plotted)
    return res_plotted

def folder_mode(model):
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    img_files.sort()
    for img_name in img_files:
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        infer_and_show(model, img)
        print(f"当前图片: {img_name}，按 q 查看下一张，按 Esc 退出。")
        key = cv2.waitKey(0)
        if key == 27:  # Esc
            break
    cv2.destroyAllWindows()

def camera_mode(model, cam_id=0):
    from MVSControl import MVSController
    cap = MVSController()
    print("按 q 退出实时推理，按 s 保存当前帧。")
    while True:
        frame = cap.get_image()
        res_plotted = infer_and_show(model, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            save_path = f"camera_save_{cv2.getTickCount()}.jpg"
            cv2.imwrite(save_path, res_plotted)
            print(f"已保存: {save_path}")
    cap.close_device()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("请选择模式：1-文件夹图片验证  2-摄像头实时推理")
    mode = input("输入1或2并回车: ").strip()
    model = YOLO(YOLO_WEIGHTS)
    if mode == "1":
        folder_mode(model)
    elif mode == "2":
        camera_mode(model)
    else:
        print("无效输入，程序退出。")