#!/usr/bin/env python3
import socket
import pickle
import queue
import os
import cv2
import threading
import numpy as np  # needed for warmup printing
from MVSControl import MVSController
from ComputePose import ImageProcessor
from ConstConfig import Const
from Transform import *

VISION_HOST = Const.Vision.HOST
VISION_PORT = Const.Vision.PORT

YOLO_WEIGHTS = os.path.join(Const.Yolo.MODEL_DIE,
                            Const.Yolo.YOLO_HOLE_WEIGHTS)
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
            self.img_processor = ImageProcessor(model_weights=YOLO_WEIGHTS, show=show, waitkey=WAITKEY)

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
            if client_command.get('command') == 'detect':
                obj = client_command.get('object')
                if obj == HOLE:
                    target_hole_idx = client_command.get('target_hole_idx', 0)
                    t1 = cv2.getTickCount()
                    img = self.mvs_handle.get_image()
                    if img is None:
                        print("未获取到图像")
                        continue
                    t3 = cv2.getTickCount()
                    hole_list, show_img = self.img_processor.detect_hole(
                        img,
                        circle_fit_method='EdgeDrawing',
                    )
                    if show_img is not None:
                        self.send_to_display(show_img)
                    
                    # 越界/无效检查
                    if (not hole_list or
                        hole_list[target_hole_idx][2] is None):
                        print("Target circle not found")
                        error_data = pickle.dumps({'status': 'error', 'message': 'Target circle not found'})
                        client_socket.sendall(len(error_data).to_bytes(4, byteorder='big'))
                        client_socket.sendall(error_data)
                        continue

                    u, v, r = hole_list[target_hole_idx]
                    t4 = cv2.getTickCount()

                    # 序列化数
                    img_bytes = self.encode_img(img)
                    result = {'status': 'success', 'result': [float(u), float(v), float(r)], 'img': img_bytes}
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
                
                elif obj == CALIB:
                    t1 = cv2.getTickCount()
                    img = self.mvs_handle.get_image()
                    if img is None:
                        print("未获取到图像")
                        continue
                    t3 = cv2.getTickCount()
                    calib_circle, show_img = self.img_processor.detect_calib_hole(
                        img,
                        circle_fit_method='EdgeDrawing',
                    )

                    if show_img is not None:
                        self.send_to_display(show_img)
                    t4 = cv2.getTickCount()

                    # 序列化数
                    u, v, r = calib_circle
                    img_bytes = self.encode_img(img)
                    result = {'status': 'success', 'result': [float(u), float(v), float(r)], 'img': img_bytes}
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
                
                elif obj == GEAR:
                    t1 = cv2.getTickCount()
                    img = self.mvs_handle.get_image()
                    if img is None:
                        print("未获取到图像")
                        continue
                    t3 = cv2.getTickCount()
                    gear_pos, gear_angle, show_img = self.img_processor.detect_gear(
                        img,
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
    client_socket , addr = vision_server.server_socket.accept()
    print(f"来自 {addr} 的连接已建立。")
    vision_server.stop_warmup()

    # 处理客户端请求
    print("开始处理客户端请求.")
    vision_server.handle_client(client_socket)

    client_socket.close()
    vision_server.close()
    print(f"与 {addr} 的连接已关闭。")

if __name__ == '__main__':
    main()
