# Reconnaissance faciale

Si vous pensez que la reconnaissance faciale est très complexe à mettre en place, détrompez-vous. Je vais vous montrer que n'importe qui peut le faire. Et c'est d'ailleurs ça qui est inquiétant. Si en important 2-3 librairies, j'ai pu faire un petit programme si facilement, imaginez ce qu'un gouvernement peut mettre en place. 

## Librairies et dépendances

Ce programme utilise plusieurs librairies, listées ci-dessous ou dans le fichier "requirement.txt". Pour installer toutes ces librairies, utilisez la commande pip suivante : 

```
pip install -r requirement.txt
```

Les librairies suivantes seront téléchargées (entre parenthèse, la commande pip pour installer spécifiquement la librairie):   
- OpenCV, spécialisé dans le traitement d'images en temps réel (**`pip install opencv-python`**, puis **`pip install opencv-contrib-python`** pour certaines fonctionnalités)
- NumPy, destinée à manipuler des matrices ou tableaux multidimensionnels, ainsi que des fonctions mathématiques opérant sur ces tableaux (**`pip install numpy`**)
- PIL, bibliothèque de traitement d’image (**`pip install Pillow`**)
- Json, un format simple de stockage et d'échange de données. Il est utlisé pour stocker les noms et IDs des personnes enregistrées. (json est un module intégré dans Python, il n'y a pas besoin de l'installer avec pip)


## Explication du programme

Lancez le fichier "main" (**`python main.py`**). Vous pourrez alors choisir entre ajouter quelqu'un à reconnaitre ou lancer la reconnaissance.

### Ajout d'une personne

Tout d'abord, le programme datasetCreator va être lancer. Son but est de localiser un visage et d'en prendre autant d'échantillons que spécifié en paramètre. Voilà comment un visage est localisé : 

``` python
ret, img = cam.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, 1.3, 5)
```

Puis on enregistre les informations de la personne scannée. 

Ensuite, le programme trainner va récupérer les images et les vectoriser. La fonction "getImagesAndLabels" permet de charger les images du dataset. La fonction "train" de OpenCV va nous permettre d'entrainer un modèle de reconnaissance faciale. 

``` python
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
```

Le visage est vectorisé et enregistré dans le fichier "trainner.yml".

### Reconnaissance de visage

Le programme detector va, comme le programme datasetCreator, chercher les visages présents. Puis, à l'aide d'un objet "LBPHFaceRecognizer",

``` python
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
```

le programme va chercher à prédire si le visage correspond aux données présentes dans l'objet "recognizer".

``` python
idPerson, conf = recognizer.predict(gray[y:y+h, x:x+w])
```

Voilà !

## Conclusion

Comme vous pouvez le voir, il n'y a rien de très compliqué. Juste un peu de programmation basique et un bon usage de fonctions déjà existantes. 


Voici, pour finir, une excellente vidéo abordant le sujet de la protection vis-à-vis de la reconnaissance faciale : https://youtu.be/tbdcL5Ux-9Y (en anglais).
