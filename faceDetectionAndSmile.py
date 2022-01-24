"""
    Visão Computacional
    Alunos: Lucas da Silva Lima, 
    Luiz Henrique Barros de Souza,
    Nicolas Fernandes Lima
    Universidade do Estado do Amazonas
    Escola Superior de Tecnologia
"""

import face_recognition
import cv2
import dlib

def faceLandmarks(im):
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
    faceRects = faceDetector(im, 0)
    landmarksAll = []
    for i in range(0, len(faceRects)):
        newRect = dlib.rectangle(int(faceRects[i].left()),
                            int(faceRects[i].top()),
                            int(faceRects[i].right()),
                            int(faceRects[i].bottom()))
        landmarks = landmarkDetector(im, newRect)
        landmarksAll.append(landmarks)

    return landmarksAll, faceRects


def renderFacialLandmarks(im, landmarks):
    
    points = []
    [points.append((p.x, p.y)) for p in landmarks.parts()]
    for p in points:
        cv2.circle(im, (int(p[0]),int(p[1])), 2, (255,0,0),-1)

    return im


video_capture = cv2.VideoCapture(0)

lucas = face_recognition.load_image_file("lucas.jpeg")
lucas_enc = face_recognition.face_encodings(lucas)[0]

luiz = face_recognition.load_image_file("luiz.jpeg")
luiz_enc = face_recognition.face_encodings(luiz)[0]

nicolas = face_recognition.load_image_file("nicolas.jpeg")
nicolas_enc = face_recognition.face_encodings(nicolas)[0]

known_face_encodings = [lucas_enc, luiz_enc, nicolas_enc]
known_face_names = ["Lucas", "Luiz", "Nicolas"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while(True):
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks, _ = faceLandmarks(gray)

    if(process_this_frame):
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        if(face_encodings):    
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            name = "Estranho"

            if (True in matches):
                first_match_index = matches.index(True)
                lips_width = abs(landmarks[0].parts()[49].x - landmarks[0].parts()[55].x)
                jaw_width = abs(landmarks[0].parts()[3].x - landmarks[0].parts()[15].x)
                ratio = lips_width/jaw_width
                if ratio > 0.32 :
                    result = "Smile"
                    name = known_face_names[first_match_index]+' '+result
                else:
                    result = "No Smile"
                    name = known_face_names[first_match_index]+' '+result

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    #press Q to quit window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
