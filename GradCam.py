import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model('hazmat_model_trained-16layer2.h5')

# Load and preprocess input image
img_path = 'radioactive-sign.png'  # Replace with the actual path to your image
img = cv2.imread(img_path)
img = cv2.resize(img, (64, 64))  # Resize to match model input size
img = img / 255.0
img_array = np.expand_dims(img, axis=0)  # Add batch dimension

# Get the class index with the highest probability
preds = model.predict(img_array)
class_index = np.argmax(preds)

# Grad-CAM algorithm
last_conv_layer_name = 'output_layer' #'last_conv_layer'  # Choose the last convolutional layer
grad_model = Model(inputs=model.input, outputs=(model.get_layer(last_conv_layer_name).output, model.output))
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_output = predictions[:, class_index]
grads = tape.gradient(class_output, conv_outputs)[0]
pooled_grads = np.mean(grads, axis=(0, 1, 2))
heatmap = np.maximum(np.sum(conv_outputs[0] * pooled_grads, axis=-1), 0)

# Normalize the heatmap
heatmap /= np.max(heatmap)

# Overlay heatmap on the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

# Convert heatmap to float32 and ensure img is also float32
heatmap = np.float32(heatmap) / 255
img = np.float32(img)

# Superimpose the heatmap on the original image
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Display the results
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
