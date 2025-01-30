import cv2
import matplotlib.pyplot as plt

def main():
    #reading an image
    img = cv2.imread("JacobFace.jpg", cv2.IMREAD_COLOR)

    #Converting to gray scale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Load Cascade file
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #Detect Faces
    faces_rect = haar_cascade.detectMultiScale(grayImg, scaleFactor= 1.1, minNeighbors=9)

    #Iterating through the rectangles of found faces
    for (x,y,w,h) in faces_rect:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
    plt.axis('off')  # Hide axis
    plt.show()



































if __name__ == "__main__":
    main()