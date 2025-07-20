from ultralytics import YOLO

import os,time
import multiprocessing as mp


current_path = os.path.dirname(__file__)

if __name__ == '__main__':
    mp.freeze_support() 

    # Load a model
    model = YOLO(f"{current_path}/weights/yolov8n-seg.pt")

    # Train the model
    t1=time.time()
    results = model.train(data=f"{current_path}\\train.yaml", epochs=200, imgsz=640, workers=4)
    t2=time.time()

    print(f"========== Training time: {t2-t1} s ==========")