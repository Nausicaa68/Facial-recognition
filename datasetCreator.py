import cv2


def dataset_creation(sampleNumber):

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    id = input("Enter an id > ")
    sampleNum = 0

    while(sampleNum < sampleNumber):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # incrementing sample number
            sampleNum += 1

            # saving the captured face in the dataset folder
            cv2.imwrite("dataSet/User." + id + '.' +
                        str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
            cv2.imshow('frame', img)

    cam.release()
    cv2.destroyAllWindows()

    name = input("Enter the name corresponding to this person > ")

    memoryFile = open('memoryFile', 'a')
    memoryFile.write(id + ":" + name + '\n')
    memoryFile.close()


if __name__ == "__main__":
    dataset_creation(30)
