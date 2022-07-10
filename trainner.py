import numpy as np
import cv2
import os
from PIL import Image


def getImagesAndLabels(path):
    # obtenir le schéma pour les images
    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    except FileNotFoundError:
        print("Error. There is no dataset.")

    # création d'une liste de visage et initialisation des IDs
    faceSamples = []
    Ids = []

    detector = cv2.CascadeClassifier(
        "haarcascade/haarcascade_frontalface_default.xml")

    # boucle pour charger les identifiants
    for imagePath in imagePaths:
        # conversion vers l'espace de niveau de gris
        pilImage = Image.open(imagePath).convert('L')
        # conversion de la liste en une liste numpy (vecteur d'images)
        imageNp = np.array(pilImage, 'uint8')
        # obtenir l'ID de chaque image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extraire la face de l'échantillon d'image d'entraînement
        faces = detector.detectMultiScale(imageNp)
        # charger la liste
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
            cv2.imshow("trainning", imageNp)
            cv2.waitKey(10)
    return faceSamples, Ids


def vectorTrainner():

    if(os.path.exists("./DataSet") == True):
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces, Ids = getImagesAndLabels('DataSet')
        try:
            recognizer.train(faces, np.array(Ids))
        except:
            print("Error ! You need at least more than one sample to learn a model")

        recognizer.save('trainner/trainner.yml')
        cv2.destroyAllWindows()
    else:
        print("Error. There is no dataset folder.")


if __name__ == "__main__":
    vectorTrainner()
