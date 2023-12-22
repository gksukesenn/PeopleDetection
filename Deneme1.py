from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

cap = cv2.VideoCapture(0)      # for webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../YOLO-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
mask = cv2.imread("mask1.png", cv2.IMREAD_GRAYSCALE)



detections = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, img, mask=mask)
    results = model(imgRegion, stream=True)
    cv2.rectangle(img, (0, 0), (1280, 720), (0, 255, 0), 5)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if mask[int((y1 + y2) / 2), int((x1 + x2) / 2)] == 255:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == "person" and conf > 0.3:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    currentArray = np.array([x1, y1, x2, y2])
                    detections.append(currentArray)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                    cv2.rectangle(img, (280, 86), (992, 643), (0, 255, 255), 3)



    cv2.imshow("Image", img)
    cv2.waitKey(1)
