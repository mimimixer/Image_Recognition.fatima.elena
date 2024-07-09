from keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('hazmat_model_trained.h5')

# Load and preprocess input image
img_path = 'radioactive-sign.png'  # Replace with the actual path to your image
img = cv2.imread(img_path)
img = cv2.resize(img, (32, 32))  # Resize to match training data dimensions
img = img / 255.0  # Normalize pixel values to [0, 1]
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make prediction
preds = model.predict(img)
class_index = np.argmax(preds, axis=-1)
print("Predicted class index:", class_index)

# If you have class labels, you can map the index to the actual class name
class_names = ["Poison Warning", "Oxygen", "Flammable", "Corrosive", "Dangerous for Environment", "Non-2 gas", "Explosive", "Radioactive", "Inhalation", "Biohazard"]  # Replace with your class names
if 0 <= class_index < len(class_names):
    predicted_class = class_names[class_index[0]]  # Access the scalar value from the array
    print("Predicted class:", predicted_class)
else:
    print("Invalid class index:", class_index)


