import numpy as np
import cv2
import pickle
from keras.models import load_model

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.60
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
model = load_model('hazmat_model_trained.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocess(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getClassName(classNumber):
    if classNumber == 0:
        return 'Poison Warning'
    elif classNumber == 1:
        return 'Oxygen'
    elif classNumber == 2:
        return 'Flammable'
    elif classNumber == 3:
        return 'Corrosive'
    elif classNumber == 4:
        return 'Dangerous for Environment'
    elif classNumber == 5:
        return 'Non-2 gas'
    elif classNumber == 6:
        return 'Explosive'
    elif classNumber == 7:
        return 'Radioactive'
    elif classNumber == 8:
        return 'Inhalation'
    elif classNumber == 9:
        return 'Biohazard'

while True:
    success, imgOriginal = cap.read()

    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocess(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOriginal, "Class: ", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability: ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        cv2.putText(imgOriginal, str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break