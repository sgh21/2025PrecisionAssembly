import cv2
import os, sys
import numpy as np
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("workspace:", workspace)
sys.path.append(workspace)
sys.path.append(os.path.join(workspace, 'configs'))
from ultralytics import YOLO
from ConstConfig import Const

IMG_SHAPE_SHOW = Const.Camera.IMG_SHAPE_SHOW
YOLO_WEIGHTS = os.path.join(workspace, Const.Yolo.MODEL_DIR, Const.Yolo.YOLO_HOLE_WEIGHTS)
DATASET_DIR = Const.Data.DATASET_DIR
# IMG_DIR = os.path.join(DATASET_DIR, 'yolo_0803/train/images')
IMG_DIR = os.path.join(workspace, 'test', 'images')

def infer_and_show(model, img, win_name="YOLO"):
    results = model(img)
    res_plotted = results[0].plot()
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, IMG_SHAPE_SHOW[1], IMG_SHAPE_SHOW[0])
    cv2.imshow(win_name, res_plotted)
    return res_plotted

    # results = model.predict(img, retina_masks = True, conf = Const.Yolo.YOLO_CONF)
    # result_img = img.copy()

    # # 遍历检测结果
    # boxes = results[0].boxes.xyxy.cpu().numpy()
    # masks = results[0].masks.data.cpu().numpy()
    # classes = results[0].boxes.cls.cpu().numpy().astype(int)
    # # probs = results[0].probs.cpu().numpy()
    # for i in range(len(boxes)):

    #     cls_id = int(classes[i])  # 类别ID
    #     mask = masks[i] if masks is not None else None  # 获取掩码
    #     # conf = probs[i]  # 置信度
    #     x1, y1, x2, y2 = map(int, boxes[i])  # 边界框坐标

    #     # 如果有掩码，绘制自定义颜色
    #     if mask is not None:
    #         mask_resized = (mask > 0.5).astype(np.uint8)  # 二值化掩码
    #         mask_resized = np.expand_dims(mask_resized, axis=-1)  # 扩展为 (H, W, 1)
    #         mask_resized = np.repeat(mask_resized, 3, axis=-1)    # 复制通道，变为 (H, W, 3)

    #         # 自定义颜色（例如：gear 类别为绿色）
    #         if cls_id == Const.ClassInfo.GEAR_CLASS or cls_id == Const.ClassInfo.CLASS_DICT[Const.ClassInfo.GEAR_CLASS]:
    #             color = (0, 255, 0)  # 绿色
    #         else:
    #             color = (255, 0, 0)  # 默认蓝色
    #         # color = np.array(color, dtype=np.uint8)

    #         # 在原图上绘制掩码
    #         colored_mask = np.zeros_like(result_img, dtype=np.uint8)
    #         print(colored_mask.shape, mask_resized.shape)
    #         result_img = np.where(mask_resized == 1,
    #                                 result_img * 0.9 + color * 0.1,
    #                                 result_img)
    #         # result_img = cv2.addWeighted(result_img, 0.7, colored_mask, 0.3, 0)

    #     # 绘制边界框和类别标签
    #     label = f"{Const.ClassInfo.CLASS_NAME_DICT[cls_id]}"# {conf:.2f}"
    #     cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
    #     cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示结果
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, IMG_SHAPE_SHOW[1], IMG_SHAPE_SHOW[0])
    cv2.imshow(win_name, result_img)
    return result_img

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