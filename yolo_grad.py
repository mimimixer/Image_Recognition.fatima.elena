import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torch.autograd import Function

# Ensure GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the Grad-CAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook the target layer to get the gradients
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        # Forward pass
        logits = self.model(x)

        # Zero grads
        self.model.zero_grad()

        # Backward pass
        class_score = logits[0, class_idx]
        class_score.backward()

        # Get the activations and gradients from the target layer
        activations = self.target_layer.output
        gradients = self.gradients

        # Compute the weights
        weights = torch.mean(gradients, dim=[0, 2, 3])

        # Compute the Grad-CAM heatmap
        grad_cam = torch.zeros(activations.shape[2:]).cuda()
        for i, w in enumerate(weights):
            grad_cam += w * activations[0, i, :, :]

        grad_cam = torch.relu(grad_cam)
        grad_cam = grad_cam / torch.max(grad_cam)
        grad_cam = grad_cam.cpu().data.numpy()

        return grad_cam

# Load the YOLOv8 model
model = YOLO("best.pt")
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Identify the target layer
target_layer = model.model[0][-2]  # Adjust the index based on the model architecture

# Initialize Grad-CAM
grad_cam = GradCAM(model, target_layer)

# Load and preprocess the input image
img_path = 'radioactive-sign.png'
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))  # Resize to match model input size
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)
img = torch.tensor(img).float().to(device)

# Get the class index (you might need to adjust this part based on your use case)
with torch.no_grad():
    preds = model(img)
class_idx = torch.argmax(preds[0])

# Generate Grad-CAM heatmap
heatmap = grad_cam(img, class_idx)

# Display the heatmap
plt.imshow(heatmap, cmap='jet')
plt.axis('off')
plt.show()

# Overlay heatmap on the original image
heatmap = cv2.resize(heatmap, (img.shape[3], img.shape[2]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Convert the original image to BGR format
img = img[0].cpu().numpy().transpose(1, 2, 0) * 255
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Superimpose the heatmap on the original image
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Display the results
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
