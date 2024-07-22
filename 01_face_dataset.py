import cv2

# Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = input("\n Enter user id :") 
print ("\n [INFO] Initializing face capture. Look at the camera and wait ...")
count = 0

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use the default camera (0) or specify the camera index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Look for faces in the image using the loaded cascade file
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    print("Found "+str(len(faces))+" face(s)")
    
    # Draw a rectangle around every found face
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(x, y, w, h)
    
    # Save the result image
    if len(faces):
        count += 1
        img_item = "F:\Downloads\opencv_facerecognition\opencv_facerecognition\dataset/User." + str(face_id) + '.' + str(count) + ".jpg"
        cv2.imwrite(img_item, roi_gray)
    
    # Display a frame    
    cv2.imshow("Frame", frame)
    
    # Wait for 'q' key to be pressed and break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if count == 2000:
        break
camera.release()
cv2.destroyAllWindows()
