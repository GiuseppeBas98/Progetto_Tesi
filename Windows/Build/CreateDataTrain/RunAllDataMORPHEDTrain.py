import os
import time

def esegui_file(file_path):
    print(f"Esecuzione del file: {file_path}")
    os.system(f"python {file_path}")

def main():
    # Lista dei percorsi dei tuoi 8 file

    for i in range(1, 9):
        percorso_cartella = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\ma_cropped\SubFolder_Morphed_" + str(i)
        with open("Cartella.txt", "w") as file:
            file.truncate(0)  # Elimina il contenuto esistente
            file.write(percorso_cartella)

        esegui_file("DataLoaderMorphedTrain.py")
        print("Fine 2k Foto.\nAttendi 15sec prima che le prossime 2k foto vengono eseguite... :)")
        time.sleep(10)  # Ritardo di 20 secondi tra le esecuzioni

if __name__ == "__main__":
    main()
