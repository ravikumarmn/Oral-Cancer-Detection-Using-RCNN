from flask import jsonify, render_template, Flask, request
from flask_cors import CORS, cross_origin
from flask_caching import Cache
import os
import cv2
import numpy as np
import torch
import torchvision
import os
import cv2
import numpy as np
import torch
import torchvision
from flask_caching import Cache
from flask_cors import CORS, cross_origin
from predict import plantleaf
from PIL import Image

# Initialize the Flask app and configure cache
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})



def load_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load('maskrcnn_resnet50_fpn.pt', map_location=torch.device('cpu')))
    model.eval()
    return model


def process_image(img, model):
    
    # Use the Mask R-CNN model to get the masks, boxes, and labels
    with torch.no_grad():
        outputs = model([torch.from_numpy(img).permute(2, 0, 1).float()])
    masks = outputs[0]['masks'].cpu().numpy()
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    # Create a color map for the predicted labels
    colors = [[0, 255, 0], [255, 0, 0]]

    # Loop over the predicted masks, boxes, and labels to create segmented masks and bounding boxes
    for i, mask in enumerate(masks):
        # Convert the predicted mask to binary mask
        mask = np.squeeze(mask)
        binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        binary_mask[np.where(mask > 0.5)] = 255

        # Apply the binary mask to the image
        x1, y1, x2, y2 = boxes[i].astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
        if x1 >= x2 or y1 >= y2:
            continue
        binary_mask = cv2.resize(binary_mask[y1:y2, x1:x2], (x2 - x1, y2 - y1))
        segmented_mask = cv2.bitwise_and(img[y1:y2, x1:x2], img[y1:y2, x1:x2], mask=binary_mask)

        # Draw the bounding box around the segmented mask
        if labels[i] < len(colors):
            color = colors[labels[i]]
        else:
            color = [np.random.randint(0, 256) for _ in range(3)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Overlay the segmented mask onto the input image
        img[y1:y2, x1:x2] = cv2.addWeighted(segmented_mask, 0.5, img[y1:y2, x1:x2], 0.5, 0)

    # Save the output image to the output directory
    output_path = os.path.join("APP/static/images/box_image/", "image.jpg")
    cv2.imwrite(output_path, img)



@app.route("/", methods=['GET'])
@cross_origin()
def home():
    if request.method == "GET":
        return render_template('index.html')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        image_file = request.files['file']
        # check if the file is empty
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        # check if the file is an allowed format
        if not allowed_file(image_file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        # read the image file as numpy array
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # perform prediction
        classifier = plantleaf()  # creating a object of plantleaf class
        result = classifier.predictPlantImage(image_file)
        
        # return prediction result as JSON response
        return jsonify(result)
    else:
        return jsonify({"error": "Method not allowed"}), 405

@app.route('/run_classifier', methods=['POST','GET'])
@cache.cached(timeout=60*60, query_string=True)
def run_classifier():
    return """
        <img src="APP/static/images/box_image/image.jpg" alt="Oral cancer image">
        """
if __name__ == "__main__":
    app.run(debug=True,port=8080)


