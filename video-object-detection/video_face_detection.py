import numpy as np
import face_recognition as fr
import cv2

video = cv2.VideoCapture(0)

image_select = fr.load_image_file("images/pp.jpg")
image_select_encoding = fr.face_encodings(image_select)[0]

# Neymar
image_select_neymar = fr.load_image_file("images/neymar.jpg")
image_select_neymar_encoding = fr.face_encodings(image_select_neymar)[0]

# Data Base
face_encodings_know = [image_select_encoding, image_select_neymar_encoding]
face_names_know = ["Kristhyan", "Neymar"]

while True:
    conectado, frame = video.read()
    frame_rgb = frame[:, :, ::-1]

    face_locations = fr.face_locations(frame_rgb)
    face_encodings = fr.face_encodings(frame_rgb, face_locations)
    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):

        matches = fr.compare_faces(face_encodings_know, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(face_encodings_know, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = face_names_know[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 155), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 225, 200), 1)

    cv2.imshow("Detection Face", frame)

    if cv2.waitKey(1) == ord("x"):
        break

video.release()
cv2.destroyAllWindows()
