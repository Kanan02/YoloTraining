from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="C:/Users/Kenan/PycharmProjects/YoloTraining/datasets/my_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        project="runs",
        name="detect-train",
    )

    print("Training completed!")
    print("Check out:", results)

if __name__ == "__main__":
    main()
