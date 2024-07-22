import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = "F:\Downloads\opencv_facerecognition\opencv_facerecognition\dataset"

# Use Local Binary Patterns Histograms
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Please wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write(r"C:\Users\ADMIN-PC\OneDrive\Máy tính\New folder (2)\trainer.yml")  # recognizer.save() works on most platforms

# Print the number of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting program".format(len(np.unique(ids))))
