from ultralytics import YOLO

model = YOLO("model_v2/YOLO26n_buoy_detector_v2.pt")

metrics = model.val(split='test', device='cpu')
print(metrics.box.map)

