# Face Detection

# importing the libraries
import cv2

# loading the haar cascades
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

border_color = (255, 255, 255)

def draw_boundary(frame, x, y, w, h, border_color):
    # Top left border
    cv2.line(frame, (x, y), (x + (w//5), y), border_color, 2)
    cv2.line(frame, (x, y), (x, y+(h//5)), border_color, 2)

    # Top right border
    cv2.line(frame, (x+((w//5)*4), y), (x+w, y), border_color, 2)
    cv2.line(frame, (x+w, y), (x+w, y+(h//5)), border_color, 2)

    # Bottom left border
    cv2.line(frame, (x, (y+(h//5*4))), (x, y+h), border_color, 2)
    cv2.line(frame, (x, (y+h)), (x + (w//5), y+h), border_color, 2)

    # Bottom right border
    cv2.line(frame, (x+((w//5)*4), y+h), (x + w, y + h), border_color, 2)
    cv2.line(frame, (x+w, (y+(h//5*4))), (x+w, y+h), border_color, 2)

def detect(frame):
    
    # Convert frame into gray scale image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces and store the positions
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbours=5)
    
    for x, y, w, h in faces:
        draw_boundary(frame, x, y, w, h, border_color)
        
    return frame
        
        