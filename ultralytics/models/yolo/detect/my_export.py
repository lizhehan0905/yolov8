# from ultralytics import YOLOv10

# model = YOLOv10("runs/detect/train_yolov10s_tank+tk/weights/best.pt")

# results = model.export(format="onnx", simplify=True)


from ultralytics import YOLOv10

model = YOLOv10("yolov10s.pt")

results = model.export(format="onnx", simplify=True,opset=13)