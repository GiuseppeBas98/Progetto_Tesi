import glob

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tqdm import tqdm
from tqdm import tqdm



def augmentation(img_path,dir,prefix,iteration):
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


def start():
    for index in tqdm(range(0, 8), desc="Extracting images"):
        dir = "D:\\PiscopoRoberto\\FER\\CK+PreProcessed\\"
        save_dir="D:\\PiscopoRoberto\\FER\\CK+PreProcessedAugmented\\"
        iterations=5
        match index:
            case 0:
                dir = dir + "HAPPY"
                iterations = 6
                save_dir = save_dir + "HAPPY"
            case 1:
                dir = dir + "NEUTRAL"
                iterations = 4
                save_dir = save_dir + "NEUTRAL"
            case 2:
                dir = dir + "ANGER"
                iterations = 9
                save_dir = save_dir + "ANGER"
            case 3:
                dir = dir + "DISGUST"
                iterations = 7
                save_dir = save_dir + "DISGUST"
            case 4:
                dir = dir + "FEAR"
                iterations = 15
                save_dir = save_dir + "FEAR"
            case 5:
                dir = dir + "SADNESS"
                iterations = 12
                save_dir = save_dir + "SADNESS"
            case 6:
                dir = dir + "SURPRISE"
                iterations = 5
                save_dir = save_dir + "SURPRISE"
            case 7:
                dir = dir + "CONTEMPT"
                save_dir = save_dir +"CONTEMPT"
                iterations = 20



        for filename in glob.glob(dir + '/*.*'):
                fotoName = filename.split(dir + '\\')[1]
                subject = fotoName.split("_")[0]
                augmentation(dir+'\\'+fotoName,save_dir, subject,iterations)

#start()

dir = "D:\\PiscopoRoberto\\FER\\CK+PreProcessed\\SURPRISE"
save_dir="D:\\PiscopoRoberto\\FER\\CK+PreProcessedAugmented\\SURPRISE"
iterations = 5

for filename in glob.glob(dir + '/*.*'):
    fotoName = filename.split(dir + '\\')[1]
    subject = fotoName.split("_")[0]
    augmentation(dir + '\\' + fotoName, save_dir, subject, iterations)
