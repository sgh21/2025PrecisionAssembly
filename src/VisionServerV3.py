'''
粗定位方案：
Yolo: hole, gear
SAM: keyhole
'''

#!/usr/bin/env python3
import socket
import pickle
import queue
import torch
import cv2
import threading
import numpy as np  # needed for warmup printing
import os, sys
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("workspace:", workspace)
sys.path.append(workspace)
sys.path.append(os.path.join(workspace, 'configs'))
from MVSControl import MVSController
from ComputePoseV3 import ImageProcessor
from ConstConfig import Const
from Transform import *

device = "cuda" if torch.cuda.is_available() else "cpu"

VISION_HOST = Const.Vision.HOST
VISION_PORT = Const.Vision.PORT

YOLO_WEIGHTS = os.path.join(workspace, Const.Yolo.MODEL_DIR,
                            Const.Yolo.YOLO_HOLE_WEIGHTS)

SAM_MODEL_TYPE = Const.Sam.SAM_MODEL_TYPE
SAM_WEIGHTS = os.path.join(workspace, Const.Sam.MODEL_DIR, Const.Sam.SAM_WEIGHTS)

WAITKEY = Const.Task.WAITKEY  # OpenCV窗口等待时间

TARGET_HOLE_IDX_LIST = Const.Task.TARGET_HOLE_IDX_LIST 

# CLASS INFO
GEAR = Const.ClassInfo.GEAR_CLASS
HOLE = Const.ClassInfo.HOLE_CLASS
CALIB = Const.ClassInfo.CALIB_CLASS
def recv_exact(sock: socket.socket, n: int) -> bytes:
    """阻塞读取正好 n 字节；若对端关闭或异常则抛出 EOFError"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf.extend(chunk)
    return bytes(buf)

class VisionServer:
    def __init__(self, host = VISION_HOST, port = VISION_PORT, show = True):
        self.show = show
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许快速复用端口
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        self._stop_warmup = False

        # 显示线程相关
        self._stop_display = False
        self.img_queue = queue.Queue(maxsize=5)
        self.display_thread = None
        self.warmup_thread = None

        print(f"Vision server started on {host}:{port}")

    def init_camera(self, show = True):
        try:
            self.mvs_handle = MVSController()
            # *： 实例化图片处理类
            self.img_processor = ImageProcessor(
                device=device, yolo_model_weights=YOLO_WEIGHTS,
                model_weights=SAM_WEIGHTS, model_type=SAM_MODEL_TYPE, show=show, waitkey=WAITKEY)

            return True
        except Exception as e:
            print(f"Camera init failed: {e}")
            return False
    
    def display_worker(self):
        """专用显示线程，负责从队列读取图像并显示"""
        if not self.show:
            return
            
        print("Display thread started")
        cv2.namedWindow("Vision Processing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vision Processing", 
                        Const.Camera.IMG_SHAPE_SHOW[1], 
                        Const.Camera.IMG_SHAPE_SHOW[0])
        
        while not self._stop_display:
            try:
                # 从队列获取图像，超时1秒
                img_data = self.img_queue.get(timeout=1.0)
                if img_data is None:  # 结束信号
                    break
                    
                cv2.imshow("Vision Processing", img_data)
                key = cv2.waitKey(WAITKEY) & 0xFF
                if key == 27:  # ESC退出
                    print("ESC pressed, stopping display")
                    self._stop_display = True
                    break
                    
            except queue.Empty:
                continue
                
        cv2.destroyAllWindows()
        print("Display thread stopped")
    
    def send_to_display(self, img):
        """非阻塞发送图像到显示队列"""
        if not self.show or img is None:
            return
        try:
            self.img_queue.put_nowait(img)
        except queue.Full:
            # 队列满时丢弃最旧的图像
            try:
                self.img_queue.get_nowait()
                self.img_queue.put_nowait(img)
            except queue.Empty:
                pass
    @staticmethod
    def encode_img(img):
        """将 numpy 图像编码为基础格式（bytes），适合网络传输"""
        # 推荐用 PNG，无损且体积小
        success, buffer = cv2.imencode('.png', img)
        if not success:
            return None
        return buffer.tobytes()

    def handle_client(self,  client_socket: socket.socket):
        self.img_to_detect = None
        while True:
            # 接收客户端的请求：4字节长度 + payload
            try:
                data_len_bytes = recv_exact(client_socket, 4)
            except EOFError:
                print("客户端已断开连接（读取长度失败）")
                break

            try:
                data_len = int.from_bytes(data_len_bytes, byteorder='big')
                data = recv_exact(client_socket, data_len)
            except EOFError:
                print("客户端已断开连接（读取数据失败）")
                break

            try:
                client_command = pickle.loads(data)
            except Exception as e:
                print(f"反序列化失败：{e}")
                break
            
            # 处理指令
            print(f"收到指令：{client_command.get('command')}")
            if client_command.get('command') == 'capture':
                t1 = cv2.getTickCount()
                img = self.mvs_handle.get_image()
                while img is None:
                    print("未获取到图像，重新拍照")
                    img = self.mvs_handle.get_image()
                    cv2.waitKey(20)
                self.img_to_detect = img
                self.send_to_display(img)
                t2 = cv2.getTickCount()
                print(f"拍照时间：{(t2 - t1) / cv2.getTickFrequency()}s")
                result = {'status': 'success'}
                data_to_send = pickle.dumps(result)
                client_socket.sendall(len(data_to_send).to_bytes(4, byteorder='big'))
                client_socket.sendall(data_to_send)

            elif client_command.get('command') == 'detect':
                obj = client_command.get('object')
                if obj == HOLE:
                    target_hole_idx = client_command.get('target_hole_idx', 0)
                    if isinstance(target_hole_idx, int):
                        idx_list = [target_hole_idx]
                    elif isinstance(target_hole_idx, (list, tuple, np.ndarray)):
                        idx_list = list(target_hole_idx)
                    t1 = cv2.getTickCount()
                    hole_list, show_img = self.img_processor.detect_hole(
                        self.img_to_detect,
                        circle_fit_method='EdgeDrawing',
                    )
                    if show_img is not None:
                        self.send_to_display(show_img)
                    
                    results = []
                    for idx in idx_list:
                        if idx < 0 or idx >= len(hole_list) or hole_list[idx][2] is None:
                            results.append(None)
                        else:
                            u, v, r = hole_list[idx]
                            results.append([float(u), float(v), float(r)])

                    t2 = cv2.getTickCount()

                    # 序列化数
                    img_bytes = self.encode_img(img)
                    result = {'status': 'success', 'result': results, 'img': img_bytes}
                    data_to_send = pickle.dumps(result)
                    # 发送数据长度
                    client_socket.sendall(len(data_to_send).to_bytes(4, byteorder='big'))
                    # 发送数据
                    client_socket.sendall(data_to_send)
                    print('处理结果已发送给客户端')
                    t3 = cv2.getTickCount()
                    print(f'YOLO处理时间：{(t2 - t1) / cv2.getTickFrequency()}s')
                    print(f'数据传输时间：{(t3 - t2) / cv2.getTickFrequency()}s')

                elif obj == CALIB:
                    t1 = cv2.getTickCount()
                    calib_circle, show_img = self.img_processor.detect_calib_hole(
                        self.img_to_detect,
                        circle_fit_method='EdgeDrawing',
                    )

                    if show_img is not None:
                        self.send_to_display(show_img)
                    t4 = cv2.getTickCount()

                    # 序列化数
                    img_bytes = self.encode_img(img)
                    if calib_circle is not None:
                        u, v, r = calib_circle
                        result = {'status': 'success', 'result': [float(u), float(v), float(r)], 'img': img_bytes}
                    else:
                        result = {'status': 'error', 'result': None, 'img': img_bytes}
                    data_to_send = pickle.dumps(result)
                    # 发送数据长度
                    client_socket.sendall(len(data_to_send).to_bytes(4, byteorder='big'))
                    # 发送数据
                    client_socket.sendall(data_to_send)
                    print('处理结果已发送给客户端')
                    t2 = cv2.getTickCount()
                    print(f'YOLO处理时间：{(t4 - t1) / cv2.getTickFrequency()}s')
                    print(f'数据传输时间：{(t2 - t4) / cv2.getTickFrequency()}s')
                
                elif obj == GEAR:
                    t1 = cv2.getTickCount()
                    gear_pos, gear_angle, show_img = self.img_processor.detect_gear(
                        self.img_to_detect,
                        circle_fit_method='EdgeDrawing',
                    )
                    if show_img is not None:
                        self.send_to_display(show_img)
                    if gear_pos is None or gear_angle is None:
                        print("Target gear not found")
                        error_data = pickle.dumps({'status': 'error', 'message': 'Target gear not found'})
                        client_socket.sendall(len(error_data).to_bytes(4, byteorder='big'))
                        client_socket.sendall(error_data)
                        continue
                    t4 = cv2.getTickCount()

                    # 序列化数
                    img_bytes = self.encode_img(img)
                    result = {'status': 'success', 'result': [gear_pos, float(gear_angle)], 'img': img_bytes}
                    data_to_send = pickle.dumps(result)
                    # 发送数据长度
                    client_socket.sendall(len(data_to_send).to_bytes(4, byteorder='big'))
                    # 发送数据
                    client_socket.sendall(data_to_send)
                    print('处理结果已发送给客户端')
                    t2 = cv2.getTickCount()
                    print(f'get_image处理时间：{(t3 - t1) / cv2.getTickFrequency()}s')
                    print(f'YOLO处理时间：{(t4 - t3) / cv2.getTickFrequency()}s')
                    print(f'数据传输时间：{(t2 - t4) / cv2.getTickFrequency()}s')

                else:
                    # 未知 object
                    error_data = pickle.dumps({'status': 'error', 'message': f"Unknown object: {obj}"})
                    client_socket.sendall(len(error_data).to_bytes(4, byteorder='big'))
                    client_socket.sendall(error_data)

            elif client_command.get('command') == 'exit':
                print("收到 exit 指令，关闭连接。")
                break
            else:
                error_data = pickle.dumps({'status': 'error', 'message': f"Unknown command: {client_command.get('command')}"})
                client_socket.sendall(len(error_data).to_bytes(4, byteorder='big'))
                client_socket.sendall(error_data)


    def warmup(self):
        print("Start warming up.")

        while not self._stop_warmup:
            img = self.mvs_handle.get_image()
            if img is None:
                print("未获取到图像")
                continue
            gear_pos, hole_list, gear_angle, calib_hole, show_img = self.img_processor.process_image(img, circle_fit_method='EdgeDrawing')
            
            if show_img is not None:
                self.send_to_display(show_img)

            print(f"Gear position: {gear_pos}")
            print(f"Hole list: {hole_list}")
            print(f"Gear angle: {gear_angle * 180 / np.pi:.2f} degrees")

        print("Stop warming up")

    def start(self):
        self._stop_warmup = False
        self._stop_display = False
        # 启动显示线程
        if self.show:
            self.display_thread = threading.Thread(target=self.display_worker)
            self.display_thread.daemon = True  # 设置为守护线程
            self.display_thread.start()
        
        self.warmup_thread = threading.Thread(target=self.warmup)
        self.warmup_thread.daemon = True  # 设置为守护线程
        self.warmup_thread.start()

    def stop_warmup(self):
        self._stop_warmup = True
        if self.warmup_thread:
            self.warmup_thread.join()
            self.warmup_thread = None
    
    def stop_display(self):
        self._stop_display = True
        if self.display_thread:
            self.display_thread.join()
            self.display_thread = None

    def close(self):
        self.stop_warmup()
        self.stop_display()
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        print("Vision server closed.")
    
def main():
    # 创建 VisionServer 实例 
    # * :确定是否使用相机
    vision_server = VisionServer(host = VISION_HOST, port = VISION_PORT, show=True)
    if(not vision_server.init_camera(show=False)):
        raise Exception("Camera init failed")

    # 等待客户端连接,阻塞，开始热身
    vision_server.start()

    while True:
        client_socket , addr = vision_server.server_socket.accept()
        print(f"来自 {addr} 的连接已建立。")
        vision_server.stop_warmup()

        # 处理客户端请求
        print("开始处理客户端请求.")
        try:
            vision_server.handle_client(client_socket)
        except Exception as e:
            print(f"处理客户端时发生异常: {e}")
        finally:
            client_socket.close()
            print(f"与 {addr} 的连接已关闭。")

            
def main_offline():
    mvs_handle = MVSController()
    img_processor = ImageProcessor(
                device=device, yolo_model_weights=YOLO_WEIGHTS,
                model_weights=SAM_WEIGHTS, model_type=SAM_MODEL_TYPE, show=True, waitkey=WAITKEY)
    while True:
        img = mvs_handle.get_image()
        if img is None:
            print("未获取到图像")
            continue
        gear_pos, hole_list, gear_angle, calib_hole, show_img = img_processor.process_image(img, circle_fit_method='EdgeDrawing')
        show_img = cv2.resize(show_img, (Const.Camera.IMG_SHAPE_SHOW[1], Const.Camera.IMG_SHAPE_SHOW[0]))
        cv2.imshow("Offline Processing", show_img)
        print(f"Gear position: {gear_pos}")
        print(f"Hole list: {hole_list}")
        print(f"Gear angle: {gear_angle * 180 / np.pi:.2f} degrees")
        cv2.waitKey(50)


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1 and args[1] == "offline":
        print("运行离线模式")
        main_offline()  # 调用离线模式的 main 函数
    else:
        print("运行在线模式")
        main()  # 调用在线模式的 main 函数
