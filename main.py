import datasetCreator
import trainner
import detector


def main():

    go = True
    while(go):
        print("\nQue voulez vous faire : ")
        print("  1 - Ajouter quelqu'un")
        print("  2 - Reconnaitre")
        print("  99 - Quitter ce menu")

        choice = int(input("Votre choix > "))

        if choice == 1:
            datasetCreator.dataset_creation(50)
            print("Données ajoutées. Entrainement en cours ...")
            trainner.vectorTrainner()
            print("Entrainement terminé.")
        elif choice == 2:
            detector.facialRecognition()
            print("Stop.")
        elif choice == 99:
            go = False
            print("Bye !")
        else:
            print("Erreur de choix.")


if __name__ == "__main__":
    main()
