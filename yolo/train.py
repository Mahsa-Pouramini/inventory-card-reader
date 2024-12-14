# test

from ultralytics import YOLO

def main():
    model = YOLO("yolov8l.pt")
    data_yaml = '/hdd/datasets/yolo/schmuck.yaml'
    model.train(data=data_yaml, epochs=50, cos_lr=True)

    results = model.val()

if __name__ == '__main__':
    main()