import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64
import io
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS

# --------------------------------------
# Define the CNN Model Architecture
# --------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the CNN model weights
num_classes = 9  # Adjust based on your training configuration
cnn_model = SimpleCNN(num_classes=num_classes)
cnn_model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
cnn_model.eval()  # Set the model to evaluation mode

# Define the transforms (same as used during testing/training)
img_height, img_width = 224, 224
test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Function to predict the class using the CNN model for a given PIL image crop
def predict_cnn(pil_image):
    image = test_transform(pil_image)
    image = image.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = cnn_model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# --------------------------------------
# Initialize Flask App and YOLO Model
# --------------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the YOLO detection model (using an ONNX model here)
yolo_model = YOLO("best.onnx", task='detect')

@app.route('/api/detect', methods=['POST'])
def detect_weeds():
    # Check if an image is provided in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400

    file = request.files['image']
    image_bytes = file.read()
    
    # Decode the image from bytes
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image."}), 400

    # Run YOLO detection on the image
    results = yolo_model(img, imgsz=640, conf=0.5)
    
    # Get the annotated image from YOLO for visualization
    annotated_img = results[0].plot()

    # Initialize lists to store YOLO labels and CNN predictions
    detected_labels = []
    cnn_predictions = []

    if results[0].boxes is not None:
        # Extract YOLO labels (each label corresponds to a detected object)
        detected_labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls.tolist()]
        detected_labels = list(set(detected_labels))
        
        # Extract bounding box coordinates (assumed to be in [x1, y1, x2, y2] format)
        boxes = results[0].boxes.xyxy
        # If boxes is a tensor, convert it to numpy array
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
            
        for bbox in boxes:
            # Get integer coordinates for the bounding box
            x1, y1, x2, y2 = bbox.astype(int)
            # Ensure coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            # Crop the detected region from the original image
            crop = img[y1:y2, x1:x2]
            # Convert the cropped region from BGR (OpenCV) to RGB (PIL)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            
            # Predict the class using the CNN model
            cnn_class = predict_cnn(pil_crop)
            cnn_predictions.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "predicted_class": cnn_class
            })

    # Encode the annotated image to base64 so it can be returned in JSON
    success, buffer = cv2.imencode('.jpg', annotated_img)
    if not success:
        return jsonify({"error": "Failed to encode image."}), 500
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return the combined results
    return jsonify({
        "annotated_image": annotated_base64,
        "detected_labels": detected_labels,
        "cnn_predictions": cnn_predictions
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
