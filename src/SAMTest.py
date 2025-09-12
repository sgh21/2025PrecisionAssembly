import cv2
import numpy as np
import os, sys
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("workspace:", workspace)
sys.path.append(workspace)
sys.path.append(os.path.join(workspace, 'configs'))

from segment_anything import sam_model_registry, SamPredictor
from ConstConfig import Const

IMG_SHAPE_SHOW = Const.Camera.IMG_SHAPE_SHOW
SAM_TYPE = Const.Sam.SAM_MODEL_TYPE
SAM_WEIGHTS = os.path.join(workspace, Const.Sam.MODEL_DIR, Const.Sam.SAM_WEIGHTS)
DATASET_DIR = Const.Data.DATASET_DIR
IMG_DIR = os.path.join(workspace, 'test', 'images')

def infer_and_show(predictor, img, win_name="SAM", point_coords=None, point_labels=None):
    """
    SAM推理并可视化函数
    """
    # 设置图像到预测器
    predictor.set_image(img)
    
    # 如果没有提供点坐标，使用图像中心点作为提示
    if point_coords is None:
        h, w = img.shape[:2]
        point_coords = np.array([[w//2, h//2]])
        point_labels = np.array([1])
    
    # 预测分割掩码
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # 选择分数最高的掩码
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    best_score = scores[best_mask_idx]
    
    # 可视化结果
    result_img = visualize_sam_result(img, best_mask, point_coords, point_labels, best_score)
    
    # 显示结果
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, IMG_SHAPE_SHOW[1], IMG_SHAPE_SHOW[0])
    cv2.imshow(win_name, result_img)
    
    return result_img, masks, scores

def visualize_sam_result(image, mask, points, labels, score):
    """可视化SAM分割结果"""
    result = image.copy()
    
    # 创建彩色掩码覆盖层
    colored_mask = np.zeros_like(image)
    colored_mask[mask] = [0, 255, 0]  # 绿色掩码
    
    # 添加半透明掩码覆盖
    result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)
    
    # 绘制掩码轮廓
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # 绘制提示点
    for i, point in enumerate(points):
        label = labels[i] if labels is not None else 1
        color = (255, 0, 0) if label == 1 else (0, 0, 255)  # 蓝色前景，红色背景
        cv2.circle(result, tuple(point.astype(int)), 8, color, -1)
        cv2.circle(result, tuple(point.astype(int)), 12, (255, 255, 255), 2)
    
    # 添加置信度文本
    cv2.putText(result, f'Score: {score:.3f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result

def folder_mode_interactive(predictor):
    """文件夹模式 - 支持交互式点击分割"""
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    img_files.sort()
    
    for img_name in img_files:
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
            
        # 存储点击的点
        click_points = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal click_points
            
            if event == cv2.EVENT_LBUTTONDOWN:  # 左键添加前景点
                click_points.append([x, y, 1])  # [x, y, label] 1=前景
                print(f"添加前景点: ({x}, {y})")
                display_points()
                
            elif event == cv2.EVENT_RBUTTONDOWN:  # 右键添加背景点
                click_points.append([x, y, 0])  # [x, y, label] 0=背景
                print(f"添加背景点: ({x}, {y})")
                display_points()
        
        def display_points():
            """在原图上显示点击的点"""
            display_img = img.copy()
            for point in click_points:
                x, y, label = point
                color = (255, 0, 0) if label == 1 else (0, 0, 255)  # 蓝色前景，红色背景
                cv2.circle(display_img, (x, y), 8, color, -1)
                cv2.circle(display_img, (x, y), 12, (255, 255, 255), 2)
            
            # 添加点数统计
            fg_count = sum(1 for p in click_points if p[2] == 1)
            bg_count = sum(1 for p in click_points if p[2] == 0)
            cv2.putText(display_img, f'前景点: {fg_count}, 背景点: {bg_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_img, '按空格键进行分割', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Original", display_img)
        
        def perform_segmentation():
            """执行分割"""
            if click_points:
                coords = np.array([[p[0], p[1]] for p in click_points])
                labels = np.array([p[2] for p in click_points])
                result_img, _, _ = infer_and_show(predictor, img, "SAM Result", coords, labels)
                return result_img
            else:
                print("请先设置提示点！")
                return None
        
        # 设置窗口和回调
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original", IMG_SHAPE_SHOW[1], IMG_SHAPE_SHOW[0])
        cv2.setMouseCallback("Original", mouse_callback)
        
        # 初始显示
        cv2.imshow("Original", img)
        
        print(f"当前图片: {img_name}")
        print("操作说明:")
        print("- 左键点击: 添加前景点 (蓝色)")
        print("- 右键点击: 添加背景点 (红色)")
        print("- 按空格键: 执行分割")
        print("- 按 'c': 清除所有点")
        print("- 按 'u': 撤销最后一个点")
        print("- 按 'q': 下一张图片")
        print("- 按 Esc: 退出程序")
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):  # 下一张
                break
            elif key == 27:  # Esc 退出
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):  # 空格键执行分割
                print("执行分割，请等待...")
                perform_segmentation()
            elif key == ord('c'):  # 清除所有点
                click_points.clear()
                print("清除所有点击点")
                cv2.imshow("Original", img)
                try:
                    cv2.destroyWindow("SAM Result")
                except Exception as e:
                    print("WARNING: ", e)
            elif key == ord('u') and click_points:  # 撤销最后一个点
                removed = click_points.pop()
                print(f"撤销点: ({removed[0]}, {removed[1]})")
                display_points()
        
        try:
            cv2.destroyWindow("Original")
            cv2.destroyWindow("SAM Result")
        except Exception as e:
            print("WARNING: ", e)

    cv2.destroyAllWindows()

def camera_mode_interactive(predictor):
    """摄像头模式 - 支持交互式点击分割"""
    from MVSControl import MVSController
    cap = MVSController()
    
    click_points = []
    current_frame = None
    is_segmented = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal click_points
        
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键添加前景点
            click_points.append([x, y, 1])
            print(f"添加前景点: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键添加背景点
            click_points.append([x, y, 0])
            print(f"添加背景点: ({x}, {y})")
    
    def display_frame_with_points(frame):
        """在帧上显示点击点"""
        display_frame = frame.copy()
        for point in click_points:
            x, y, label = point
            color = (255, 0, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_frame, (x, y), 8, color, -1)
            cv2.circle(display_frame, (x, y), 12, (255, 255, 255), 2)
        
        # 添加状态信息
        fg_count = sum(1 for p in click_points if p[2] == 1)
        bg_count = sum(1 for p in click_points if p[2] == 0)
        cv2.putText(display_frame, f'前景点: {fg_count}, 背景点: {bg_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, '按空格键分割 | F冻结画面', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return display_frame
    
    def perform_segmentation():
        """执行分割"""
        nonlocal is_segmented
        if click_points and current_frame is not None:
            coords = np.array([[p[0], p[1]] for p in click_points])
            labels = np.array([p[2] for p in click_points])
            result_img, _, _ = infer_and_show(predictor, current_frame, "SAM Camera", coords, labels)
            is_segmented = True
            return result_img
        else:
            print("请先设置提示点！")
            return None
    
    print("摄像头交互模式启动:")
    print("- 左键点击: 添加前景点")
    print("- 右键点击: 添加背景点")
    print("- 按空格键: 执行分割")
    print("- 按 'f': 冻结/解冻画面")
    print("- 按 's': 保存当前结果")
    print("- 按 'c': 清除所有点")
    print("- 按 'u': 撤销最后一个点")
    print("- 按 'q' 或 Esc: 退出")
    
    cv2.namedWindow("SAM Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SAM Camera", IMG_SHAPE_SHOW[1], IMG_SHAPE_SHOW[0])
    cv2.setMouseCallback("SAM Camera", mouse_callback)
    
    frozen = False
    
    while True:
        if not frozen:
            frame = cap.get_image()
            if frame is None:
                continue
            current_frame = frame.copy()
            is_segmented = False
        
        # 显示带有点击点的图像
        display_frame = display_frame_with_points(current_frame)
        cv2.imshow("SAM Camera", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 退出
            break
        elif key == ord(' '):  # 空格键执行分割
            result = perform_segmentation()
        elif key == ord('f'):  # 冻结/解冻画面
            frozen = not frozen
            status = "冻结" if frozen else "解冻"
            print(f"画面{status}")
        elif key == ord('s') and is_segmented:  # 保存
            save_path = f"sam_interactive_save_{cv2.getTickCount()}.jpg"
            # 获取分割窗口的图像
            try:
                result_window = cv2.getWindowImageRect("SAM Camera")
                print(f"已保存当前分割结果")
            except:
                cv2.imwrite(save_path, display_frame)
                print(f"已保存: {save_path}")
        elif key == ord('c'):  # 清除所有点
            click_points.clear()
            is_segmented = False
            print("清除所有点击点")
        elif key == ord('u') and click_points:  # 撤销最后一个点
            removed = click_points.pop()
            print(f"撤销点: ({removed[0]}, {removed[1]})")
    
    cap.close_device()
    cv2.destroyAllWindows()

def folder_mode(model):
    """原来的文件夹模式（使用中心点）"""
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
    """原来的摄像头模式（使用中心点）"""
    from MVSControl import MVSController
    cap = MVSController()
    print("按 q 退出实时推理，按 s 保存当前帧。")
    while True:
        frame = cap.get_image()
        res_plotted, _, _ = infer_and_show(model, frame)
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
    print("请选择模式：")
    print("1 - 文件夹图片验证 (中心点)")
    print("2 - 摄像头实时推理 (中心点)")
    print("3 - 文件夹图片验证 (交互式点击)")
    print("4 - 摄像头实时推理 (交互式点击)")
    
    mode = input("输入1-4并回车: ").strip()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_WEIGHTS).to(device=device)
    model = SamPredictor(sam)
    
    if mode == "1":
        folder_mode(model)
    elif mode == "2":
        camera_mode(model)
    elif mode == "3":
        folder_mode_interactive(model)
    elif mode == "4":
        camera_mode_interactive(model)
    else:
        print("无效输入，程序退出。")