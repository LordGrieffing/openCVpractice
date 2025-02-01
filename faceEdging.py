import cv2
import numpy as np


def main():
    
    #Start webcam
    cap = cv2.VideoCapture(0)

    #Load the Haar model
    kernal = np.ones((3,3), np.float32)/30
    #kernal = np.array([[-1,-1,-1], [0,0,0], [1,2,1]])
    #kernal = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])


    
    while True:

        #capture frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        #run the kernal over the frame
        frame = cv2.filter2D(frame,-1, kernal)

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