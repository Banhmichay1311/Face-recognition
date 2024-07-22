import cv2

# initialize the camera
camera = cv2.VideoCapture(1)  # Use the default camera (0) or specify the camera index if you have multiple cameras

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Adjust the path to your trained model file

# Load the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Adjust the path to your cascade file

font = cv2.FONT_HERSHEY_SIMPLEX

# Names related to IDs
names = ['none', 'Vu Ngoc Tram', 'Ly','Dat','4','5','6','7','8','Kien'] 

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Look for faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    print("Found " + str(len(faces)) + " face(s)")

    # Draw a rectangle around every found face and predict the person
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(roi_gray)

        # Check if confidence is less than 100
        if confidence < 60:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
