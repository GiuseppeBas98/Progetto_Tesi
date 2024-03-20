import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm

def augmentation(img_path, dir, prefix, iteration):
    datagen = ImageDataGenerator(
            rotation_range=10,
            horizontal_flip=True,
            )

    img = load_img(img_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    i=1
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=dir, save_prefix=prefix, save_format='png'):
        i += 1
        if i > iteration:
            break  # otherwise the generator would loop indefinitely


def start(dir, save_dir, iterations):
    file_list = os.listdir(dir)
    for filename in tqdm(file_list, desc="Elaborazione Immagini"):
        percorso_immagine = os.path.join(dir, filename)
        foto_name = percorso_immagine.split(os.path.sep)[-1]
        augmentation(percorso_immagine, save_dir, foto_name, iterations)


dir = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\bf_cropped\SubFolder_Bonafide_8"
save_dir = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\bf_cropped\SubFolder_Bonafide_9"
iterations = 4 #PRIMA 10 E POI 2
start(dir, save_dir, iterations)