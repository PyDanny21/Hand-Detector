
import cv2
import mediapipe as mp
import math
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
import mediapipe.python.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 456)
detector = HandDetector(detectionCon=0.8, maxHands=2)
face = FaceDetector(minDetectionCon=2)

def main():
    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)
        # faces=face.findFaces(img)
        # detector.findPosition(img,draw=False)
        # Display
        cv2.imshow("Image", img)
        c=cv2.waitKey(1)
        if c==ord('q'):
            break
if __name__ == '__main__':
    main()