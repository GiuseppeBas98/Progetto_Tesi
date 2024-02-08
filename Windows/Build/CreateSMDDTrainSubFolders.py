import os
import shutil

def divide_in_sottocartelle_M(cartella_origine, cartella_destinazione, foto_per_sottocartella):
    # Creazione della cartella di destinazione se non esiste
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)

    # Conta le foto nella cartella di origine
    foto_totali = sorted(os.listdir(cartella_origine))

    # Crea le sottocartelle e copia le foto
    for i, foto in enumerate(foto_totali):
        sottocartella_numero = i // foto_per_sottocartella + 1
        sottocartella = os.path.join(cartella_destinazione, f'SubFolder_Morphed_{sottocartella_numero}')

        if not os.path.exists(sottocartella):
            os.makedirs(sottocartella)

        origine_foto = os.path.join(cartella_origine, foto)
        destinazione_foto = os.path.join(sottocartella, foto)
        shutil.copy(origine_foto, destinazione_foto)

def divide_in_sottocartelle_BF(cartella_origine, cartella_destinazione, foto_per_sottocartella):
    # Creazione della cartella di destinazione se non esiste
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)

    # Conta le foto nella cartella di origine
    foto_totali = sorted(os.listdir(cartella_origine))

    # Crea le sottocartelle e copia le foto
    for i, foto in enumerate(foto_totali):
        sottocartella_numero = i // foto_per_sottocartella + 1
        sottocartella = os.path.join(cartella_destinazione, f'SubFolder_Bonafide_{sottocartella_numero}')

        if not os.path.exists(sottocartella):
            os.makedirs(sottocartella)

        origine_foto = os.path.join(cartella_origine, foto)
        destinazione_foto = os.path.join(sottocartella, foto)
        shutil.copy(origine_foto, destinazione_foto)

if __name__ == "__main__":
    cartella_origineM = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\ma_cropped\Morphed"
    cartella_destinazioneM = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\ma_cropped"
    cartella_origineBF = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\bf_cropped\Bonafide"
    cartella_destinazioneBF = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\bf_cropped"
    foto_per_sottocartella = 2000

    divide_in_sottocartelle_M(cartella_origineM, cartella_destinazioneM, foto_per_sottocartella)
    divide_in_sottocartelle_BF(cartella_origineBF, cartella_destinazioneBF, foto_per_sottocartella)
