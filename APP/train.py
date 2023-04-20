from flask import Flask, Response, render_template, request
import torch
import torchvision
import numpy as np
import cv2

# Define the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the classes
classes = ['background', 'oral_cancer']

# Define a function to detect oral cancer and draw bounding boxes
def detect_oral_cancer(image):
    # Convert the image to a tensor
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # Run the image through the model
    outputs = model([image_tensor])

    # Get the bounding boxes and scores for the predictions
    boxes = outputs[0]['boxes'].detach().numpy()
    scores = outputs[0]['scores'].detach().numpy()

    # Filter out predictions with a low score
    threshold = 0.5
    boxes = boxes[scores >= threshold]
    scores = scores[scores >= threshold]

    # Draw the bounding boxes on the image
    for box in boxes:
        box = box.astype(int)
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Return the image with the bounding boxes
    return image

# Define the Flask app
app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    if request.method == "GET":
        return render_template('ml.html')
    
# Define a route to upload an image and run the oral cancer detector on it
@app.route('/detect_oral_cancer', methods=['POST'])
def detect_oral_cancer_route():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return 'No file uploaded'

    # Get the uploaded image file
    file = request.files['file']

    # Read the image file
    image_bytes = file.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Run the oral cancer detector on the image
    output_image = detect_oral_cancer(image)

    # Convert the output image to bytes
    _, output_image_bytes = cv2.imencode('.jpg', output_image)
    output_image_bytes = output_image_bytes.tobytes()

    # Return the output image
    return Response(output_image_bytes, mimetype='image/jpeg')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug = True,port = 2022)
