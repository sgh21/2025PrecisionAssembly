# 测试硬件通讯
1. 使用MVSControl.py测试相机获取图像是否显示正确
2. 使用AuboControlLowLevel.py测试机器人通讯

:::success
测试机器人运动时，一定注意初始位姿，防止撞坏相机

:::

# 确定相机平面、拍照和标定位置
> 为了尽可能减少标定需要的时间，保持测试环境和比赛环境一致，应该使得机器人末端距离齿轮安装板的距离和测试时一致
>



1. 将机器人引导至插入位置平面，然后使用AuboControlLowLevel.py的代码片段进行Z轴标定
2. 打开MVSControl.py分别将机器人移动至keyhole中心和calib_circle中心，确定拍照位置和标定位置，并将相关位姿写入ConstConfig.py文件

# 测试视觉算法，并采集图像微调
> 主要测试Yolo的检测结果是否符合预期
>

1. 运行YoloTest.py并观察测试结果
2. Yolo检测效果不好则使用DataCollector.py进行图片采集，并标注，使用YoloTrain.py进行训练
3. 使用VisionServer.py进行整体视觉算法结果测试

# 内参标定和外参标定
1. 同时运行标定程序（IntrinsicCalib.py）和视觉服务器，对内参进行标定，并写入ConstConfig.py文件
2. 修改ConstConfig.py文件的TARGET_HOLE_IDX确定标定孔，运行HoleInsert.py进行预插孔，使用示教器步进模式微调，使得其插入孔内，并抬起机械臂，放入齿轮。
3. 运动HandInEyeCalib.py和视觉服务器进行手眼标定，并将相关结果写入HAND_IN_EYE_OFFSET

# 全流程插孔测试
1. 修改ConstConfig.py文件的TARGET_HOLE_IDX_LIST确定标定孔并运行RobotClientV2.py和视觉服务器，进行全流程插孔测试

# Yolo补训和更新
1. 使用YoloTrain.py对Yolo模型补训
2. 并修改ConstConfig.py文件的YOLO_HOLE_WEIGHTS为新模型名称

