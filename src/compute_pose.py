from ultralytics import YOLO
from scipy.ndimage import binary_erosion
import torch
import numpy as np
import cv2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO("../yolo/runs/segment/train/weights/best.pt")  # Load a trained model

class Const:
    IMG_SHAPE_O = (448, 640, 3)  # YOLO输出图像尺寸
    GEAR_Z = 18                  # 齿轮的齿数

class SegmentResult:
    '''存储一个分割对象的信息的类
    后缀为i的变量表示拍摄图像的属性，尺寸[2048,3072,3]
    后缀为o的变量表示模型输出的图像属性，尺寸[448,640,3]
    后缀为l的变量表示局部坐标系的属性
    '''
    class_dict = {
        0: "gear",      # 齿轮外轮廓
        1: "hole",      # 安装孔洞
        2: "keyhole"    # 齿轮内轮廓（齿轮键槽）
    }
    def __init__(self, img, class_id, box, mask: np.ndarray):
        self.position = None    # 位姿，待计算
        self.radius = None      # 半径，待计算
        self.angle = None       # 角度，待计算
        self.img_i = img
        self.img_o = cv2.resize(img, (640, 448))  # 输出图像尺寸
        self.class_id = class_id
        self.class_name = self.class_dict[class_id]
        self.box_i = box
        self.mask = mask
        self.scale_io = Const.IMG_SHAPE_O[0]/img.shape[0]  # scale_io<1，输出图像和输入图像的缩放比例
        self._cut_pic_with_box(padding = 30)
        self._calc_bound_with_mask()    # 方法1：yolo输出的mask边界
        self._calc_bound_with_box()     # 方法2: 用局部图像进行canny得到的边界
        self._filter_edge_point()       # 用mask边界过滤canny边界
    
    def _calc_bound_with_mask(self)->None:
        '''使用掩码计算边界点'''
        kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)
        eroded_mask = binary_erosion(self.mask, structure=kernel)    # 四边全1的掩码

        edge_mask = self.mask.astype(bool) & ~eroded_mask
        # print("Edge mask shape:", edge_mask.shape)
        self.mask_edge_point_o = np.column_stack(np.where(edge_mask))
    
    def _cut_pic_with_box(self, padding:int) -> None:
        '''根据边界框裁剪图片'''
        x1, y1, x2, y2 = self.box_i.astype(int)
        height, width= self.img_i.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width - 1, x2 + padding)
        y2 = min(height - 1, y2 + padding)
        self.xyxy_i = (x1, y1, x2, y2)
        self.xyxy_o = (int(x1 * self.scale_io), int(y1 * self.scale_io),
                       int(x2 * self.scale_io), int(y2 * self.scale_io))
        self.local_img_i = self.img_i[y1:y2, x1:x2]
    
    def _calc_bound_with_box(self) -> None:
        '''在局部图像上使用Canny边缘检测
        '''
        gray = cv2.cvtColor(self.local_img_i, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v = np.median(gray)
        if self.class_name == "gear":
            sigma = 0.1
        elif self.class_name == "hole":
            sigma = 0.33
        elif self.class_name == "keyhole":
            sigma = 0.5
        low_threshold = int(max(0, (1.0 - sigma) * v))
        high_threshold = int(min(255, (1.0 + sigma) * v))
    
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edge_point = np.column_stack(np.where(edges > 0))
        self.edge_point_l = edge_point  # 局部坐标系
        self.edge_point_i = edge_point + np.array([self.xyxy_i[1], self.xyxy_i[0]])  # 转换为原图坐标系

    def _filter_edge_point(self, min_distance=5)-> None:
        '''由mask边界过滤canny边界，排除canny边界中离mask边界距离大于阈值的点
        '''
        # print(self.mask_edge_point_o, self.edge_point_i)
        valid_points = []
        for p_i in self.edge_point_i:
            # p_i = p + np.array([self.xyxy_i[1], self.xyxy_i[0]])
            if np.any(np.linalg.norm(self.mask_edge_point_o / self.scale_io - p_i, axis=1) < min_distance):
                valid_points.append(p_i)
        self.edge_point_i = np.array(valid_points)

    def draw_edge(self,img=None, thickness=2, local=True) -> np.ndarray:
        '''
        在原图上绘制边界点
        输出：局部图尺寸/网络输出图尺寸 绘制边界后的图像
        '''
        if img is None:
            img = self.img_o.copy()
        if local:   # 输出为局部图像尺寸
            img = self.local_img_i.copy()
            for point in self.edge_point_l:
                cv2.circle(img, tuple(point[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img
        else:       # 输出为网络输出图尺寸
            for p in self.mask_edge_point_o:    # 绘制mask边界
                cv2.circle(img, tuple(p[::-1]), radius=1, color=(255, 0, 0), thickness=thickness)
            for p in self.edge_point_i:         # 绘制canny边界
                # p_o = (p + np.array([self.xyxy_i[1], self.xyxy_i[0]]))* self.scale_io
                p_o = p * self.scale_io
                p_o = p_o.astype(int)
                cv2.circle(img, tuple(p_o[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img

def filter_keyhole_cicle(seg:SegmentResult) -> list[list[int,int]]:
    '''对于keyhole类(齿轮内轮廓)，额外将bbox纵向2/3以下部分的点删去
    '''
    valid_points = []
    if seg.class_name == "keyhole":
        valid_points = seg.edge_point_i # 获取边缘点列表
        # 对于keyhole类，排除bbox纵向2/3以下部分的点
        y_threshold = seg.xyxy_i[1] + (seg.xyxy_i[3] - seg.xyxy_i[1]) * 2 / 3
        valid_points = [p for p in valid_points if p[0] < y_threshold]
    return valid_points

def locate_circle(points:list[list[int,int]])-> tuple[float, float, float]:
    '''
    使用代数最小二乘法拟合圆心和半径
    返回: (center_x, center_y, radius), 原图坐标系
    '''
    if len(points) < 3:
        return None, None, None
    points = np.array(points, dtype=float)
    x = points[:, 1]
    y = points[:, 0]
    n = len(x)
    # 计算质心
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # 中心化坐标
    u = x - x_mean
    v = y - y_mean
    # 构建系数矩阵
    Suu = np.sum(u * u)
    Suv = np.sum(u * v)
    Svv = np.sum(v * v)
    Suuu = np.sum(u * u * u)
    Suvv = np.sum(u * v * v)
    Svvv = np.sum(v * v * v)
    Suuv = np.sum(u * u * v)
    # 求解线性系统
    A = np.array([[Suu, Suv],
                  [Suv, Svv]])
    b = np.array([0.5 * (Suuu + Suvv),
                  0.5 * (Svvv + Suuv)])
    uc, vc = np.linalg.solve(A, b)
    # 转换回原坐标系
    center_x = uc + x_mean
    center_y = vc + y_mean
    # 计算半径
    radius = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / n)
    return center_x, center_y, radius

def locate_all_segment(seg_list:list[SegmentResult], img=None, debug=False) -> tuple[tuple[int,int,int],list[tuple[int,int,int]],float]:
    '''综合处理所有分割对象，输出：齿轮和6个孔的位置+半径，以及齿轮的旋转角度
    '''
    gear_pos = (0, 0)   # 齿轮位置，原图坐标系
    hole_pos_list = []  # 孔洞位置列表，原图坐标系
    gear_angle = None   # 范围 [0, 2*pi/z]
    gear_list = []
    keyhole_list = []
    hole_list = []
    for seg in seg_list:
        if seg.class_name == "gear":
            gear_list.append(seg)
        elif seg.class_name == "hole":
            hole_list.append(seg)
        elif seg.class_name == "keyhole":
            keyhole_list.append(seg)
            
    # 计算齿轮的角度
    if len(gear_list) != 1:
        print(f"WARNING: Gear Detection Failed. Detected Number:{len(gear_list)}")
    for seg in gear_list:
        pass # TODO: 计算齿轮的角度
    # 计算齿轮的位置
    if len(keyhole_list) != 1:
        print(f"WARNING: Keyhole Detection Failed. Detected Number:{len(keyhole_list)}")
    for seg in keyhole_list:
        filtered_points = filter_keyhole_cicle(seg)
        cx, cy, r = locate_circle(filtered_points)

        # if debug and img is not None: # 绘制keyhole边界
        #     for p in filtered_points:
        #         img = cv2.circle(img, tuple(p[::-1]), radius=1, color=(0, 255, 0), thickness=2)
        #     print(f"filtered keyhole points: {len(filtered_points)}")
        #     img = cv2.resize(img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
        #     cv2.imshow("Keyhole Circle", img)
        #     cv2.waitKey(0)
        if cx is not None and cy is not None and r is not None:
            gear_pos= (int(cx), int(cy),int(r))
    # # 使用霍夫圆变换更精确地定位齿轮位置 -> 效果不佳
    # for seg in keyhole_list:
    #     limg = seg.local_img_i.copy()
    #     gray = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (9, 9), 2)
    #     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
    #                                param1=100, param2=50, minRadius=50, maxRadius=300)
    #     if circles is not None:
    #         for circle in circles[0, :]:
    #             cx, cy, r = circle
    #             ci = np.array([seg.xyxy_i[1], seg.xyxy_i[0]])+np.array([cx, cy])  # 转换为原图坐标系
    #             if np.linalg.norm(ci - np.array([gear_pos[1],gear_pos[0]])) > 10:  # 限制在齿轮位置附近
    #                 continue
    #             if debug:
    #                 limg = cv2.circle(limg, (int(cx), int(cy)), radius=int(r), color=(0, 255, 0), thickness=2)
    #                 print(f"Detected circle: center=({cx}, {cy}), radius={r}") 
    #         if limg is not None and debug:
    #             cv2.imshow("Hough Circle Detection", limg)
    #             cv2.waitKey(0)
        
    if len(hole_list) != 6:
        print(f"Warning: Hole Detection Failed. Detected Number:{len(hole_list)}")
    radius_list = []
    for seg in hole_list:
        cx, cy, r = locate_circle(seg.edge_point_i)
        if cx is not None and cy is not None and r is not None:
            hole_pos_list.append((int(cx), int(cy),int(r)))
            radius_list.append(r)
    if len(hole_list) > 6:
        # 保留半径靠近均值的6个孔洞
        mean = np.mean(radius_list)
        radius_err = [abs(r - mean) for r in radius_list]
        min_index = np.argsort(radius_err)[:6]  
        hole_pos_list = [hole_pos_list[i] for i in min_index]
    return gear_pos, hole_pos_list, gear_angle
    
def main():
    img = cv2.imread("img1.jpg")    
    img = img[:img.shape[0],:int(img.shape[0]*Const.IMG_SHAPE_O[1]/Const.IMG_SHAPE_O[0])] # 裁剪使得两方向缩放比例相同
    print(f"Image shape: {img.shape}")
    # 使用YOLO模型进行预测
    results = model.predict(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    seg_list:list[SegmentResult] = []
    for i in range(len(boxes)):
        seg_list.append(SegmentResult(
            img = img,
            class_id = classes[i].item(),
            box = boxes[i],
            mask = masks[i]
        ))
    print(f"Detected {len(seg_list)} segments.")

    show_img = img.copy()
    # 绘制位置检测结果
    gear_pos, hole_pos_list, gear_angle = locate_all_segment(seg_list, img=show_img, debug=False)
    show_img = cv2.circle(show_img, gear_pos[:2], radius=5, color=(0, 0, 255), thickness=5)
    show_img = cv2.circle(show_img, gear_pos[:2], radius=gear_pos[2], color=(0, 0, 255), thickness=5)
    for hole_pos in hole_pos_list:
        show_img = cv2.circle(show_img, hole_pos[:2], radius=5, color=(0, 0, 255), thickness=5)
        show_img = cv2.circle(show_img, hole_pos[:2], radius=hole_pos[2], color=(0, 0, 255), thickness=5)
    # 
    show_img = cv2.resize(show_img, (Const.IMG_SHAPE_O[1], Const.IMG_SHAPE_O[0]))
    # 绘制边缘提取结果
    # for seg in seg_list:
    #     x1, y1, x2, y2 = seg.xyxy_i
    #     print(f"{seg.class_name} bbox: ({x1}, {y1}), ({x2}, {y2})")
    #     show_img = seg.draw_edge(img=show_img, thickness=1, local=False)
    cv2.imshow("Segmented Image", show_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

