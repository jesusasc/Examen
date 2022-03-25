from ast import Break
import statistics
from turtle import width
from unittest import result
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5 ) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            Break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            for han_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, han_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Captura de mano", frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    cap.release()
    cv2.destroyAllWindows()

#Este es para el Face Mesh

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks,
                    mp_face_mesh.FACE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1))

        cv2.imshow("Captura de mano", frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break


cap.release()
cv2.destroyAllWindows()