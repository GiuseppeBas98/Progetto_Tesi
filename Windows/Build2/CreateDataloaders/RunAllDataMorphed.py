import os
import time

def esegui_file(file_path):
    print(f"Esecuzione del file: {file_path}")
    os.system(f"python {file_path}")

def main():
    # Lista dei percorsi dei tuoi 8 file

    for i in range(1, 14):
        percorso_cartella = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_FRLL_merged\Morphed_cropped\SubFolder_Morphed_" + str(i)
        with open("CartellaMerged.txt", "w") as file:
            file.truncate(0)  # Elimina il contenuto esistente
            file.write(percorso_cartella)

        esegui_file("DataloaderMorphed.py")
        print("Fine 2k Foto.\nAttendi 15sec prima che le prossime foto vengono eseguite... :)")
        time.sleep(15)  # Ritardo di 15 secondi tra le esecuzioni

if __name__ == "__main__":
    main()
