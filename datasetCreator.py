import cv2
import json


def dataset_creation(sampleNumber):

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        'haarcascade/haarcascade_frontalface_default.xml')

    idPerson = input("Enter an id (number) > ")
    sampleNum = 0

    while(sampleNum < sampleNumber):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # saving the captured face in the dataset folder
            cv2.imwrite("dataSet/User." + idPerson + '.' +
                        str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imshow('frame', img)

        # incrementing sample number
            sampleNum += 1

    cam.release()
    cv2.destroyAllWindows()

    name = input("Enter the name corresponding to this person > ")

    with open('memoryFile') as f:
        data = f.read()
    # reconstructing the data as a dictionary
    nameCorrespondance = json.loads(data)

    # adding in the dictionnary the new person data
    nameCorrespondance[idPerson] = name

    # create json object from dictionary
    jsonObj = json.dumps(nameCorrespondance)

    # open file for writing, "w", and write json object to file
    f = open("memoryFile", "w")
    f.write(jsonObj)
    f.close()


if __name__ == "__main__":
    dataset_creation(30)
