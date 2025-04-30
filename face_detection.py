import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

# Initialize the FaceMeshDetector
detector = FaceMeshDetector(maxFaces=1)

# Open the webcam (use 0 for the default camera, or replace with the path to a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Exiting...")
        break

    # Detect facial landmarks in the frame
    img, faces = detector.findFaceMesh(img, draw=True)

    # If a face is detected
    if faces:
        # Retrieve the list of 468 landmarks for the first face
        face = faces[0]

        # Example: Highlight a specific landmark (e.g., the tip of the nose, ID 1)
        nose_tip = face[1]
        cv2.circle(img, nose_tip, 5, (0, 255, 0), cv2.FILLED)

    # Display the resulting frame
    cv2.imshow("Face Landmark Tracking", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
