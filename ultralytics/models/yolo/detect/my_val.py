# from ultralytics import YOLO
 
 
# # 加载模型
# model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # 从YAML构建并转移权重
 
# if __name__ == '__main__':
#     # 训练模型
#     results = model.train(data='./mnist160', epochs=5, imgsz=64)
#     # 加上这一行代码即可进行验证了
#     model.val()


from ultralytics import YOLOv10

model = YOLOv10("/code/yolov10/ultralytics/models/yolov10/runs/detect/train_yolov10s_tank+tk/weights/best.pt")
data_yaml_path = 'my_dataset.yaml'
if __name__ == '__main__':
    results = model.val(data=data_yaml_path,
                          epochs=100,
                          batch=1,
                          imgsz=640,
                          workers=0,
                          name='val_v10')
