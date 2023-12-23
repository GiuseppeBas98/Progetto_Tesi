import os
import time

def esegui_file(file_path):
    print(f"Esecuzione del file: {file_path}")
    os.system(f"python {file_path}")

def main():
    # Lista dei percorsi dei tuoi 8 file
    for i in range(1, 9):
        percorso_cartella = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\m15k_t\SubFolder_Morphed_" + str(i)
        with open("CartellaRICCI.txt", "w") as file:
            file.truncate(0)  # Elimina il contenuto esistente
            file.write(percorso_cartella)

        esegui_file("DATAMORPHEDTrainToText.py")
        print("Fine 2k Foto.\nAttendi 20sec prima che il prossimo file viene eseguito... :)")
        time.sleep(20)  # Ritardo di 20 secondi tra le esecuzioni


    #for file_path in files_da_eseguire:
    #    esegui_file(file_path)
    #    print("Fine Ciclo.\nAttendi 20sec prima che il prossimo file viene eseguito... :)")
    #    time.sleep(20)  # Ritardo di 20 secondi tra le esecuzioni

if __name__ == "__main__":
    main()
