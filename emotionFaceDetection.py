import cv2
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch


def main():
    
    #Start webcam
    cap = cv2.VideoCapture(0)

    #Load the emotion monitor model
    processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")

    
    while True:

        #capture frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        

        # Convert OpenCV image (BGR) to PIL image (RGB)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image for the model
        inputs = processor(images=image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():  # No need to compute gradients
            outputs = model(**inputs)


        # Get predicted class
        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_label = model.config.id2label.get(predicted_class_idx, "Unknown")

        # Display the prediction on the frame
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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