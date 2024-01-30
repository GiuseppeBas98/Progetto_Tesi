import os
import time

def esegui_file(file_path):
    print(f"Esecuzione del file: {file_path}")
    os.system(f"python {file_path}")

def main():
    percorso_cartella = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\FRLL_dataset\FRLL-Morphs\facelab_london"
    # foldername contiene il percorso completo della sottocartella corrente
    # subfolders contiene una lista di sottocartelle nella cartella corrente
    # Verifica se ci sono sottocartelle
    for foldername, subfolders, files in os.walk(percorso_cartella):
        for subfolder in subfolders:
            subfolder_path = os.path.join(foldername, subfolder)
            with open("CartellaTest.txt", "a") as file:
                file.truncate(0)
                file.write(subfolder_path)
            print(subfolder_path)

            # Esegui lo script esegui_file per ciascuna sottocartella
            print("Inizio Foto MORPHED per il Dataloader di Test.")
            esegui_file('DataLoaderMorphedTest.py')
            print("Fine Foto MORPHED.\nAttendi 5sec prima che le foto BONAFIDE vengano eseguite... :)")
            time.sleep(5)
            esegui_file('DataLoaderBonafideTest.py')


if __name__ == "__main__":
    main()
