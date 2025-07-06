from ultralytics import YOLO
model = YOLO("yolo11n.pt")

model.train(data="SudokuDetection/data.yaml", epochs = 100, save=True, save_period=10)

results = model.val(data="SudokuDetection/data.yaml", save_json=True)
print(results.confusion_matrix.to_df)
