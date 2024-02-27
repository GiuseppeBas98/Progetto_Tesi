import os
import time

def esegui_file(file_path):
    print(f"Esecuzione del file: {file_path}")
    os.system(f"python {file_path}")

def main():
    # Ciclo dei percorsi delle 8 sottocartelle
    for i in range(10, 13):
        percorso_cartella = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_FRLL_merged\Bonafide_cropped\SubFolder_Bonafide_" + str(i)
        with open("CartellaMerged.txt", "w") as file:
            file.truncate(0)  # Elimina il contenuto esistente
            file.write(percorso_cartella)

        esegui_file("addBF.py")
        print("Fine Foto.\nAttendi 10sec prima che le prossime foto vengono eseguite... :)")
        time.sleep(10)  # Ritardo di 20 secondi tra le esecuzioni

if __name__ == "__main__":
    main()
