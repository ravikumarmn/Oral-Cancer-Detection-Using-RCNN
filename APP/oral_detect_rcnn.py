import os
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
import torch

# # Define your model
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

# # Save the model
# torch.save(model.state_dict(), 'maskrcnn_resnet50_fpn.pt')


# Load the pre-trained Mask R-CNN model
model = None
def load_model():
    global model
    if not model:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        model.load_state_dict(torch.load('maskrcnn_resnet50_fpn.pt'))
        model.eval()
load_model()

def process_image(image_path):
    img = cv2.imread(image_path)

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
    output_path = os.path.join("APP/static/images/box_image/", os.path.basename(image_path))
    cv2.imwrite(output_path, img)


process_image("APP/image.jpg")
