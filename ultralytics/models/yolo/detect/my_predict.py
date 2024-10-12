

from ultralytics import YOLOv10

# model = YOLOv10("runs/detect/train_yolov5m+v10head_tank+tk/weights/best.pt")
model = YOLOv10("/code/yolov10/ultralytics/models/yolov10/runs/detect/train_yolov10s_tank+tk/weights/best.pt")

results = model.predict(source="/datasets/test_video/20240624/",save=True,imgsz=640,conf=0.65,iou=0.45)

# from ultralytics.cfg import entrypoint
# arg="yolo detect predict model=runs/detect/train_yolov10s_tank+tk/weights/best.pt source=/datasets/datasets/object_detect/tank/coco/images/val/ save=False imgsz=640"

# entrypoint(arg)
