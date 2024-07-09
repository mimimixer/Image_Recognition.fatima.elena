import os
import random
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
path = "Hazmat_DataSet"
labelFile = "Hazmat_labels.csv"
batch_size_val = 32
steps_per_epoch_val = 2000
epochs_val = 50
image_Dimension = (32, 32, 3)
test_Ration = 0.2
validation_Ration = 0.2

# Load images and labels
count = 0
images = []
class_Number = []
myList = os.listdir(path)
print("Total classes found:", len(myList))
number_of_classes = len(myList)
print("Importing classes")
for x in range(len(myList)):
    my_image_list = os.listdir(os.path.join(path, str(count)))
    for y in my_image_list:
        current_image = cv2.imread(os.path.join(path, str(count), y))
        # Resize the image to your desired dimensions
        current_image = cv2.resize(current_image, (image_Dimension[0], image_Dimension[1]))
        images.append(current_image)
        class_Number.append(count)
    print(count, end=" ")
    count += 1
print(" ")

images = np.array(images)
class_Number = np.array(class_Number)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, class_Number, test_size=test_Ration)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_Ration)

# Calculate the correct steps_per_epoch value
num_samples = len(X_train)
steps_per_epoch_val = np.ceil(num_samples / batch_size_val)

print("Data Shapes")
print("Train", end=" "); print(X_train.shape, y_train.shape)
print("Validation", end=" "); print(X_validation.shape, y_validation.shape)
print("Test", end=" "); print(X_test.shape, y_test.shape)

# Ensure the images are in the correct dimensions
assert X_train.shape[0] == y_train.shape[0], "The number of images is not the same as the number of labels in training set"
assert X_validation.shape[0] == y_validation.shape[0], "The number of images is not the same as the number of labels in validation set"
assert X_test.shape[0] == y_test.shape[0], "The number of images is not equal to the number of labels in test set"
assert X_train.shape[1:] == image_Dimension, "The dimension of the training images are wrong"
assert X_validation.shape[1:] == image_Dimension, "The dimension of the validation images are wrong"
assert X_test.shape[1:] == image_Dimension, "The dimension of the test images are wrong"

data = pd.read_csv(labelFile)
print("data shapes", data.shape, type(data))

# Data visualization
number_of_samples = []
cols = 5
number_classes = number_of_classes
fig, axs = plt.subplots(nrows=number_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            number_of_samples.append(len(x_selected))

print(number_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, number_classes), number_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of Images")
plt.show()

# Preprocessing functions
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # Convert image to YUV color space
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Apply histogram equalization to the Y channel
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # Convert back to BGR color space
    return img_eq

def preprocess(img):
    img = equalize(img)
    img = img / 255
    return img

# Apply preprocessing
X_train = np.array(list(map(preprocess, X_train)))
X_validation = np.array(list(map(preprocess, X_validation)))
X_test = np.array(list(map(preprocess, X_test)))

# Ensure dimensions after preprocessing
print("Shapes after preprocessing")
print("Train", end=" "); print(X_train.shape)
print("Validation", end=" "); print(X_validation.shape)
print("Test", end=" "); print(X_test.shape)

# Reshape the data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 3)  # Change 1 to 3 for RGB
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 3)  # Change 1 to 3 for RGB
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 3)  # Change 1 to 3 for RGB

print("Shapes after reshaping")
print("Train", end=" "); print(X_train.shape)
print("Validation", end=" "); print(X_validation.shape)
print("Test", end=" "); print(X_test.shape)

# Data augmentation
data_Generation = ImageDataGenerator(width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     shear_range=0.1,
                                     rotation_range=10)

data_Generation.fit(X_train)
batches = data_Generation.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(image_Dimension[0], image_Dimension[1], 3))  # Ensure 3 channels
    axs[i].axis("off")
plt.show()

# One-hot encoding the labels
y_train = to_categorical(y_train, number_of_classes)
y_validation = to_categorical(y_validation, number_of_classes)
y_test = to_categorical(y_test, number_of_classes)

# Model definition
def create_model():
    number_of_filters = 60
    size_of_filters = (5, 5)
    size_of_filter_2 = (3, 3)
    size_of_pool = (2, 2)
    number_of_nodes = 500

    model = Sequential()
    model.add(Conv2D(number_of_filters, size_of_filters, input_shape=(image_Dimension[0], image_Dimension[1], 3), activation="relu"))
    model.add(Conv2D(number_of_filters, size_of_filters, activation="relu"))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Conv2D(number_of_filters // 2, size_of_filter_2, activation="relu"))
    model.add(Conv2D(number_of_filters // 2, size_of_filter_2, activation="relu", name="last_conv_layer"))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(number_of_nodes, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation="softmax", name="output_layer"))

    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Create and train the model
model = create_model()
print(model.summary())

history = model.fit(data_Generation.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=1)

# Plotting the results
plt.figure(1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training", "validation"])
plt.title("Loss")
plt.xlabel("epoch")

plt.figure(2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training", "validation"])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.show()

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Save the model
model.save('hazmat_model_trained.h5')
cv2.waitKey(0)



