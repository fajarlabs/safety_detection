'''
python detect.py --weights path/to/your/model.pt --source path/to/image_or_video --device cuda

'''
from ultralytics import YOLO
import cv2
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov8s.pt', help='Path to model weights')
parser.add_argument('--source', type=str, default='data/images', help='Source image or video path')
parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
args = parser.parse_args()

# Load the model
model = YOLO(args.weights)

# Run inference
def run_inference(source):
    if source.endswith(('.jpg', '.png', '.jpeg')):
        # Single image
        results = model.predict(source, conf=args.conf_thres)
        display_results(results[0])
    else:
        # Video file
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=args.conf_thres)
            display_results(results[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

# Display results
def display_results(results):
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0]
        cls = box.cls[0]
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(results.orig_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(results.orig_img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Detection', results.orig_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    run_inference(args.source)
