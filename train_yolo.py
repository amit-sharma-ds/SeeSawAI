from pathlib import Path

from ultralytics import YOLO

project_root = Path(__file__).resolve().parent
model = YOLO('yolov8n.pt')
model.train(
    data=str(project_root / 'datasets' / 'cctv_train' / 'data.yaml'),
    epochs=10,
    imgsz=640,
    batch=8,
    project=str(project_root / 'runs'),
    name='cctv_person_train',
    exist_ok=True,
)
