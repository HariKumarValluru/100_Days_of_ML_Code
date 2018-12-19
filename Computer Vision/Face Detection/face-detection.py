# Face Detection

# importing the libraries
import cv2

# loading the haar cascades
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

def detect(frame):
    
    # Convert frame into gray scale image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces and store the positions
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbours=5)
    