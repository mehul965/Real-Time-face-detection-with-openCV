import cv2

#Load the pre-trained face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    #Capture frame by Frame
    ret, frame = cap.read()

    #Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect face in the frame
    face = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30) )

    #Draw rectangle around the faces
    for(x, y, w, h) in face: 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

    #Display the resulting frame
    cv2.imshow("Face Detection", frame)

    #Break the loop when "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()