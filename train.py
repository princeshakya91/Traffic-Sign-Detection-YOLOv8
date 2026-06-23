from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO("./YOLO-TS_TT100K.yaml").load("yolov8n.pt")

    # Train the model
    model.train(
        data="./TT100K-2016.yaml", 
        epochs=200, 
        batch=16, 
        imgsz=800, 
        device='1',
        save_period=10,
        optimizer='AdamW',
        cos_lr=True,
        label_smoothing=0.1,
        mixup=0.15,
        copy_paste=0.1,
        close_mosaic=10,
        project='results',
        name='TT100K_HighRes'
    )

    # Evaluate model performance on the validation set
    metrics = model.val(data="./TT100K-2016.yaml", imgsz=640, batch=1, device='1')

    # Export the model to ONNX format
    path = model.export(format="engine", device='1', half=True, opset=12)