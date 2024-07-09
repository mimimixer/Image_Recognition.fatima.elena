import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model('hazmat_model_trained_vgg16')

# Load and preprocess input image
img_path = 'radioactive-sign.png'  # Replace with the actual path to your image
img = cv2.imread(img_path)
img = cv2.resize(img, (32, 32))  # Resize to match model input size
img = img / 255.0
img_array = np.expand_dims(img, axis=0)  # Add batch dimension

# Get the class index with the highest probability
preds = model.predict(img_array)
class_index = np.argmax(preds)

# Grad-CAM algorithm
#last_conv_layer_name = 'output_layer'  # Adjust based on the actual last conv layer name in your model
last_conv_layer = model.get_layer('block5_conv3')
#last_conv_layer = model.get_layer(last_conv_layer_name)
grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, class_index]

# Compute gradients of the top predicted class with respect to the output feature map of the last conv layer
grads = tape.gradient(loss, conv_outputs)

# Check the shape of the gradients and conv_outputs
print(f"conv_outputs shape: {conv_outputs.shape}")
print(f"grads shape: {grads.shape}")

# Ensure grads is not None and has the expected dimensions
if grads is not None and len(grads.shape) == 4:
    # Pool the gradients over all the axes leaving out the channel dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Apply ReLU to the heatmap to discard negative values
    heatmap = np.maximum(heatmap, 0)

    # Normalize the heatmap
    heatmap /= np.max(heatmap)

    # Resize the heatmap to the original image size
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
else:
    print("Error: Gradients are None or not in the expected shape.")

# Display the results
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
