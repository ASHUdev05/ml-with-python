import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_matched = False

reference_image = cv2.imread('reference.jpg')

def check_face(frame):
    global face_matched

    try:
        if DeepFace.verify(frame, reference_image.copy())['verified']:
            face_matched = True
        else:
            face_matched = False
    except ValueError:
        face_matched = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_matched:
            cv2.putText(frame, 'Face Matched', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Face Not Matched', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()