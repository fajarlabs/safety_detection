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
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

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

# Create a directory to save images if it doesn't exist
SAVE_DIR = "static"
os.makedirs(SAVE_DIR, exist_ok=True)

# Mount the static directory to serve saved images
app.mount("/static", StaticFiles(directory="static"), name="static")

model_person = YOLO()
# Load the YOLOv8 model Safety
model = YOLO("C:\\Research\\SafetyDetection\\construction-safety-1\\runs\\detect\\train\\weights\\best.pt")
# Load the YOLOv8 model Fall
model2 = YOLO("C:\\Research\\FallDetection\\runs\\detect\\train\\weights\\best.pt")

# Define the class names (update with your model's classes if needed)
class_names = model.names
# Define the class names (update with your model's classes if needed)
class_names2 = model2.names
# Define model person
# Assuming the class names are available in `class_names3`
class_names3 = model_person.names
#print(model_person.names)

# Draw BOX
# URL : https://polygonzone.roboflow.com/
polygon_set = [
    np.array([[9, 505], [12, 708], [882, 190], [731, 81], [6, 464]])
]
# Define the polygon zone (replace with your actual polygon points)
polygon_zone = np.array(polygon_set, dtype=np.int32)

# Helper function to read image from UploadFile
def read_image3(file: UploadFile):
    image_bytes = file.file.read()
    image = Image.open(BytesIO(image_bytes))
    return np.array(image)

# Helper function to draw bounding boxes
def draw_boxes3(image, results):
    for result in results:
        try :
            xyxy = result.xyxy[0].cpu().numpy().astype(int)
            conf = float(result.conf[0])
            cls = int(result.cls[0])
            if cls == 0 :
                class_name = 'person'

                # Draw the bounding box on the image
                cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{class_name} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as e :
            print("ERROR DISINI")
            print(e)
    # Draw the polygonal zone
    cv2.polylines(image, [polygon_zone], isClosed=True, color=(0, 0, 255), thickness=2)  # Red polygon
    return image

# Function to check if the box center is within the polygon
def is_in_polygon(box, polygon):
    box = box.tolist()
    # print(type(box))
    # print(type(polygon))
    # Hitung titik tengah dari bounding box
    box_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)  # Tuple (x, y)

    # Pastikan polygon adalah NumPy array 2D
    polygon = np.array(polygon, dtype=np.int32)  # Contoh: [[x1, y1], [x2, y2], ...]

    # Gunakan pointPolygonTest
    result = cv2.pointPolygonTest(polygon, tuple(box_center), False)  # Ensure box_center is passed as a tuple
    
    print(f"Point: {box_center}, Result: {result}")

    return result >= 0  # True jika di dalam atau di tepi polygon

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

# Helper function to read image
def read_image2(file: UploadFile):
    image_bytes = file.file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

# Helper function to draw bounding boxes
def draw_boxes2(image, results):
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = class_names2[cls]
        cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"
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
@app.post("/detect_ppe", tags=["Safety Detection"])
async def detect_ppe(file: UploadFile = File(...)):
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

# Endpoint for object detection
@app.post("/detect_fall", tags=["Fall Detection"])
async def detect_fall(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = read_image2(file)

        # Run inference
        results = model2.predict(image, conf=0.25)

        # Draw bounding boxes on the image
        image_with_boxes = draw_boxes2(image, results)

        # Save the image with detections
        saved_image_path = save_image(image_with_boxes)

        # Parse detection results
        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = class_names2[cls]

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

@app.post("/detect_zone", tags=["Zone Detection"])
async def detect_zone(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image = read_image3(file)

        # Run inference (replace `model_person.predict` with your actual model call)
        results = model_person.predict(image, conf=0.25)

        # Parse detection results
        detections = []
        in_area_boxes = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = class_names3[cls]
            if cls == 0 :
                # Check if the box center is within the polygonal zone
                if is_in_polygon(xyxy, polygon_set):
                    in_area_boxes.append(box)
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    })

        # Draw bounding boxes and the polygonal zone on the image
        image_with_boxes = draw_boxes3(image, in_area_boxes)

        # Save the image with detections
        saved_image_path = save_image(image_with_boxes)

        # Construct the URL for the saved image
        image_url = f"/static/{os.path.basename(saved_image_path)}"

        # Return JSON response
        return JSONResponse(content={"detections": detections, "image_url": image_url})

    except Exception as e:
        print(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Health check endpoint
@app.get("/", tags=["Base API"])
async def root():
    return {"message": "YOLOv8 Object Detection API is running"}
