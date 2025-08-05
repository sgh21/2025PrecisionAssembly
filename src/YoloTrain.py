# Description: Train a YOLOv8 model on a custom dataset
from ultralytics import YOLO

from ConstConfig import Const

data_dir =Const.Data.DATASET_DIR

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n-seg.yaml").load(weights='yolov8n-seg.pt')  # build a new model from scratch
    # model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    # 设置数据增强参数
    data_augmentation_params = {
        'augment': True,
        'mosaic': 1.0,
        'mixup': 0.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.9, #0.4
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'copy_paste': 0.0
    }
   
    # Use the model
    model.train(
        data=f"{data_dir}/yolo_0803/config.yaml", 
        epochs=100,
        batch = 32,
        workers = 12,
        **data_augmentation_params)  # train the model
    