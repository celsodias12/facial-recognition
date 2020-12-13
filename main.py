import tensorflow
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

path = "C:\\Projetos\\reconhecimento-facial-covid\\"

detector = MTCNN()
model = load_model(path + "detector.h5")

cap = cv2.VideoCapture(0)
size = (120, 120)

while True:

    ret, frame = cap.read()
    labels = []
    faces = detector.detect_faces(frame)

    people = 0

    for face in faces:

        x1, y1, w, h = face['box']
        x2, y2 = x1 + w, y1 + h

        roi = frame[y1:y2, x1:x2]

        roi = cv2.resize(roi,size)
        roi = np.reshape(roi, [1, 120,120, 3])

        if np.sum([roi])!=0:
            roi = (roi.astype('float')/255.0)

            pred = model.predict([[roi]])

            pred = pred[0]

            if pred[0] >= pred[1]:
                label = 'SEM MASCARA'
                color = (0,0,255)
                people = people + 1
                print('0')

            else:
                label = 'COM MASCARA'
                color = (0,255,0)
                print('1')

            label_position = (x1, y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,label, label_position, cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)

        else:
            cv2.putText(frame,'Nenhuma face encontrada',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

    cv2.putText(frame, "SEM MASCARA : " + str(people), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Higia Software', frame)

    key = cv2.waitKey(1)
    if key == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()