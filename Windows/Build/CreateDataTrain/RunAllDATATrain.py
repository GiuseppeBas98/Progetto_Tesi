import os
import time

def esegui_file(file_path):
    print(f"Esecuzione del file: {file_path}")
    os.system(f"python {file_path}")

def main():
    # Lista dei percorsi dei tuoi 8 file
    files_da_eseguire = ["DATATrainToText1.py",
                         "DATATrainToText2.py",
                         "DATATrainToText3.py",
                         "DATATrainToText4.py"
                         "DATATrainToText5.py",
                         "DATATrainToText6.py",
                         "DATATrainToText7.py",
                         "DATATrainToText8.py"
                         ]

    for file_path in files_da_eseguire:
        esegui_file(file_path)
        print("Fine Ciclo.\nAttendi 20sec prima che il prossimo file viene eseguito... :)")
        time.sleep(20)  # Ritardo di 20 secondi tra le esecuzioni

if __name__ == "__main__":
    main()
