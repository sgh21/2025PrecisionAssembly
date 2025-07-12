from ultralytics import YOLO

model = YOLO("best.pt")  # Load a trained model
model.export(format="onnx")  # Export the model to ONNX format