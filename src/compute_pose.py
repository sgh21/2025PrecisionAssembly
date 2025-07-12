from ultralytics import YOLO
from scipy.ndimage import binary_erosion
import torch
import numpy as np
import cv2

z = 18 # 齿数
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO("../yolo/runs/segment/train/weights/best.pt")  # Load a trained model


class SegmentResult:
    '''存储一个分割对象的信息的类
    后缀为i的变量表示拍摄图像的属性，尺寸[2048,3072,3]
    后缀为o的变量表示模型输出的图像属性，尺寸[448,640,3]'
    '''
    class_dict = {
        0: "gear",
        1: "hole",
        2: "keyhole"
    }
    def __init__(self, img, class_id, box, mask: np.ndarray):
        self.img_i = img
        self.img_o = cv2.resize(img, (640, 448))  # 输出图像尺寸
        self.class_id = class_id
        self.class_name = self.class_dict[class_id]
        self.box_i = box
        self.mask = mask
        self.scale_io = 448/img.shape[0]  # scale_io<1，输出图像和输入图像的缩放比例
        self._cut_pic_with_box(padding = 30)
        self._calc_bound_with_mask()  # 方法1：yolo输出的mask边界
        self._calc_bound_with_box()     # 方法2: 用原图像进行canny得到的边界
        self._filter_edge_point() #用mask边界过滤canny边界
    
    def _calc_bound_with_mask(self)->None:
        '''使用掩码计算边界点'''
        kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)
        eroded_mask = binary_erosion(self.mask, structure=kernel)    # 四边全1的掩码

        edge_mask = self.mask.astype(bool) & ~eroded_mask
        # print("Edge mask shape:", edge_mask.shape)
        self.mask_edge_point_o = np.column_stack(np.where(edge_mask))
    
    def _cut_pic_with_box(self, padding:int = 20) -> tuple[int, int, int, int]:
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
        gray = cv2.cvtColor(self.local_img_i, cv2.COLOR_BGR2GRAY)
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v = np.median(gray)
        if self.class_name == "gear":
            sigma = 0.1
        elif self.class_name == "hole":
            sigma = 0.33
        elif self.class_name == "keyhole":
            sigma = 0.33
        low_threshold = int(max(0, (1.0 - sigma) * v))
        high_threshold = int(min(255, (1.0 + sigma) * v))
    
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edge_point = np.column_stack(np.where(edges > 0))
        self.edge_point_l = edge_point  # 局部坐标系
        self.edge_point_i = edge_point + np.array([self.xyxy_i[1], self.xyxy_i[0]])  # 转换为原图坐标系

    def _filter_edge_point(self, min_distance=5)-> None:
        '''由mask边界过滤canny边界，排除canny边界中离mask边界距离大于阈值的点
        对于keyhole类(齿轮内轮廓)，额外将bbox纵向2/3以下部分的点删去
        '''
        # print(self.mask_edge_point_o, self.edge_point_i)
        valid_points = []
        for p_i in self.edge_point_i:
            # p_i = p + np.array([self.xyxy_i[1], self.xyxy_i[0]])
            if np.any(np.linalg.norm(self.mask_edge_point_o / self.scale_io - p_i, axis=1) < min_distance):
                valid_points.append(p_i)
        if self.class_name == "keyhole":
            # 对于keyhole类，排除bbox纵向2/3以下部分的点
            y_threshold = self.xyxy_i[1] + (self.xyxy_i[3] - self.xyxy_i[1]) * 2 / 3
            valid_points = [p for p in valid_points if p[0] < y_threshold]
        self.edge_point_i = np.array(valid_points)

    def draw_edge(self, thickness=2, local= True) -> np.ndarray:
        '''在原图上绘制边界点'''
        # 输出为局部图像尺寸
        if local:
            img = self.local_img_i.copy()
            for point in self.edge_point_l:
                cv2.circle(img, tuple(point[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img
        else:
            # 输出为网络输出图尺寸
            img = self.img_o.copy()
            for p in self.mask_edge_point_o:    # 绘制mask边界
                cv2.circle(img, tuple(p[::-1]), radius=1, color=(255, 0, 0), thickness=thickness)
            for p in self.edge_point_i:         # 绘制canny边界
                # p_o = (p + np.array([self.xyxy_i[1], self.xyxy_i[0]]))* self.scale_io
                p_o = p * self.scale_io
                p_o = p_o.astype(int)
                cv2.circle(img, tuple(p_o[::-1]), radius=1, color=(0, 255, 0), thickness=thickness)
            return img

def locate_circle(seg: SegmentResult)-> tuple[float, float, float]:
    """
    使用代数最小二乘法拟合圆心和半径
    返回: (center_x, center_y, radius), 原图坐标系
    """
    if len(seg.edge_point_i) < 3:
        return None, None, None
    
    points = seg.edge_point_i.astype(float)
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

def main():
    '''Img:[2048,3072,3], predict:[448,640]'''
    img = cv2.imread("img1.jpg")    
    img = img[:img.shape[0],:int(img.shape[0]*640/448)] # 裁剪使得两方向缩放比例相同
    print(f"Image shape: {img.shape}")
    results = model.predict(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    seg_list = []
    for i in range(len(boxes)):
        seg_list.append(SegmentResult(
            img = img,
            class_id = classes[i].item(),
            box = boxes[i],
            mask = masks[i]
        ))
    print(f"Detected {len(seg_list)} segments.")

    localdraw=False
    for seg in seg_list:
        # if seg.class_name == "hole": 
            x1, y1, x2, y2 = seg.xyxy_i
            print(f"{seg.class_name} bbox: ({x1}, {y1}), ({x2}, {y2})")
            # show_img = seg.local_img_i
            show_img = seg.draw_edge(thickness=2, local=localdraw)
            if not localdraw:
                if seg.class_name == 'hole' or seg.class_name == 'keyhole':
                    cx, cy, r = locate_circle(seg) 
                    show_img = cv2.circle(show_img, (int(cx*seg.scale_io), int(cy*seg.scale_io)), int(r*seg.scale_io), (0, 0, 255), 1)
                # show_img = cv2.resize(show_img, (640, 448))
            cv2.imshow(f"Cut Gear Image", show_img)
            cv2.waitKey(0)
            # break
    
if __name__ == "__main__":
    main()

