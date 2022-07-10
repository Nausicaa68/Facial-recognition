import json
import cv2
import numpy as np


def facialRecognition():

    detector = cv2.CascadeClassifier(
        'haarcascade/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainner/trainner.yml')
    font = cv2.FONT_HERSHEY_COMPLEX

    # importing the module json and read the data from the file
    with open('idCorresp/memoryFile') as f:
        data = f.read()
    # reconstructing the data as a dictionary
    nameCorrespondance = json.loads(data)

    continuer = True
    while(continuer):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            idPerson, conf = recognizer.predict(gray[y:y+h, x:x+w])
            try:
                idName = nameCorrespondance[str(idPerson)]
            except KeyError:
                idName = "unknown"

            cv2.putText(img, str(idName), (x, y+h), font, 1.5, (15, 0, 255))

        cv2.imshow('frame', img)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    facialRecognition()
