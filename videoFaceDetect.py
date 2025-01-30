import cv2


def main():
    
    #Start webcam
    cap = cv2.VideoCapture(0)

    #Load the Haar model
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    
    while True:

        #capture frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        #Gray the image
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Detect Faces
        faces_rect = haar_cascade.detectMultiScale(grayFrame, scaleFactor= 1.1, minNeighbors=9)
        for (x,y,w,h) in faces_rect:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)

        # Show the frame
        cv2.imshow('Real-Time Face Detection', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()




























if __name__ == "__main__":
    main()