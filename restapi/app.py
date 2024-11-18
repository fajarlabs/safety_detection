"""
Developed By Fajarlabs

"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import io
import os
from datetime import datetime

# Initialize FastAPI with custom metadata
app = FastAPI(
    title="Smart Detection API",  # Change the title here
    description="An API for object detection using YOLOv8. Upload an image and detect objects with bounding boxes.",  # Description
    version="1.0.0",  # API version
    contact={
        "name": "Fajar Labs",  # Contact name (optional)
        "email": "fajarrdp@gmail.com"  # Email (optional)
    },
)

# Mount the static directory to serve saved images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the YOLOv8 model
model = YOLO("C:\\Research\\SafetyDetection\\construction-safety-1\\runs\\detect\\train\\weights\\best.pt")

# Define the class names (update with your model's classes if needed)
class_names = model.names

# Helper function to read image
def read_image(file: UploadFile):
    image_bytes = file.file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# Helper function to draw bounding boxes
def draw_boxes(image, results):
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = class_names[cls]

        if class_name == 'no-helmet' or class_name == 'no-vest':
            # Draw red bounding box
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)        
        else :
            # Draw Green bounding box
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        
        label = f"{class_name} {conf:.2f}"

        if class_name == 'no-helmet' or class_name == 'no-vest':
            cv2.putText(image, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else :
            cv2.putText(image, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Helper function to save the image
def save_image(image):
    if not os.path.exists("static"):
        os.makedirs("static")
    filename = f"static/detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, image)
    return filename

# Endpoint for object detection
@app.post("/detect", tags=["Safety Detection"])
async def detect(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = read_image(file)

        # Run inference
        results = model.predict(image, conf=0.25)

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes(image, results)

        # Save the image with detections
        saved_image_path = save_image(image_with_boxes)

        # Parse detection results
        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = class_names[cls]

            # Add detection result
            detections.append({
                "class": class_name,
                "confidence": conf,
                "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            })

        # Construct the URL for the saved image
        image_url = f"/static/{os.path.basename(saved_image_path)}"

        # Return JSON response
        return JSONResponse(content={"detections": detections, "image_url": image_url})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Health check endpoint
@app.get("/", tags=["Base API"])
async def root():
    return {"message": "YOLOv8 Object Detection API is running"}
