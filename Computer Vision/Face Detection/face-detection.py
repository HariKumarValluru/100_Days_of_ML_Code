# Face Detection

# importing the libraries
import cv2

# loading the haar cascades
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

border_color = (242, 180, 10)

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
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    
    for x, y, w, h in faces:
        draw_boundary(frame, x, y, w, h, border_color)
        
    return frame
        
# Creating a video object for capturing the video from webcam
# cap = cv2.VideoCapture(0)
# Creating a video object for capturing the video from file
cap = cv2.VideoCapture("videos/avengers.mp4")
# getting number of frames
fps = cap.get(cv2.CAP_PROP_FPS)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream")

# infinity loop until we quit
while cap.isOpened():
    # Read the video object
    ret, frame = cap.read()
    if ret == True:
        canvas = detect(frame)
        cv2.imshow('Video', canvas)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
    
# When everything done, release the video object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()