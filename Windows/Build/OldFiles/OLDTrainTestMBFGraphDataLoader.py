import cv2
import os
from Windows.Build import mediapipeMesh as mp
import torch
import torch_geometric.data as data
from tqdm import tqdm
#import tensorflow as tf
from torch_geometric.loader import DataLoader
import glob
import psutil
import sys

train=0
test=0
val=0
tot=0


train_data = []
validation_data = []
test_data = []
total_data= []

X = []
y = 0



def normalizeNodefeatures(coordinates):
    # Initialize lists for normalized coordinates
    normalized_landmark_coordinates = []

    # Normalize each dimension separately
    for dim_coordinates in coordinates:
        min_val = min(dim_coordinates)
        max_val = max(dim_coordinates)
        normalized_dim_coordinates = [(x - min_val) / (max_val - min_val) for x in dim_coordinates]
        normalized_landmark_coordinates.append(normalized_dim_coordinates)

    return normalized_landmark_coordinates

def normalizeValues(list):
    # Calculate the minimum and maximum values
    minValue = min(list)
    maxValue = max(list)

    # Normalize values using the Min-Max formula
    list_normalized = [(el - minValue) / (maxValue - minValue) for el in list]
    return list_normalized

def graph2DataWithHog(graph, emotion, sub=None):
    landmark_coordinates = [graph.nodes[node]['pos'] for node in graph.nodes]

    hog_features = [graph.nodes[node]['hog'] for node in graph.nodes]

    landmark_coordinates_transposed = list(map(list, zip(*landmark_coordinates)))
    normalized_landmark_coordinates = list(map(list, zip(*normalizeNodefeatures(landmark_coordinates_transposed))))
    final_features = []
    for _ in range(36):
       final_features.append([])
    for i in range(36):
        for coord in normalized_landmark_coordinates[i]:
            final_features[i].append(coord)
        for hog in hog_features[i]:
            final_features[i].append(hog)

    x = torch.tensor(final_features, dtype=torch.float32)
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    edge_weights_normalized = normalizeValues(edge_weights)
    edge_weights = torch.tensor(edge_weights_normalized, dtype=torch.float32)

    y = torch.tensor(emotion, dtype=torch.float)

    sub_tensor = torch.tensor([ord(char) for char in sub])

    graph = data.Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y, subject=sub_tensor)

    return graph

def graph2DataAndSubject(graph, emotion, sub=None):
    landmark_coordinates = [graph.nodes[node]['pos'] for node in graph.nodes]

    landmark_coordinates_transposed = list(map(list, zip(*landmark_coordinates)))
    normalized_landmark_coordinates = list(map(list, zip(*normalizeNodefeatures(landmark_coordinates_transposed))))

    x = torch.tensor(normalized_landmark_coordinates, dtype=torch.float32)
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    edge_weights_normalized = normalizeValues(edge_weights)
    edge_weights = torch.tensor(edge_weights_normalized, dtype=torch.float32)

    y = torch.tensor(emotion, dtype=torch.float)

    sub_tensor = torch.tensor([ord(char) for char in sub])

    graph = data.Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y, subject=sub_tensor)

    return graph
def graph2Data(graph, type):
    # NODES
    # Extract landmark coordinates
    landmark_coordinates = [graph.nodes[node]['pos'] for node in graph.nodes]

    # Transpose the list of coordinates to have a list of coordinates per dimension
    landmark_coordinates_transposed = list(map(list, zip(*landmark_coordinates)))

    # Transpose normalized coordinates
    normalized_landmark_coordinates = list(map(list, zip(*normalizeNodefeatures(landmark_coordinates_transposed))))

    x = torch.tensor(normalized_landmark_coordinates, dtype=torch.float32)  # Tensor x with normalized coordinates
    #print(f'Shape of tensor x: {x.shape}, Node Features Tensor: {x}')

    # EDGES
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() # Tensor of graph edges
    #print(edge_index.shape)
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    edge_weights_normalized = normalizeValues(edge_weights)  # Normalize values
    edge_weights = torch.tensor(edge_weights_normalized, dtype=torch.float32)  # Tensor of edge weights/distances
    #print(edge_weights.shape)
    # print('label', emotion)
    # LABEL
    if(type == 'morphed'):
        y = 0
    elif (type == 'bonafide'):
        y = 1
    y = torch.tensor(y, dtype=torch.long)
    #print(y)
    #print(f'y: {y}, y_shape: {y.shape}')
    #y = F.one_hot(y, num_classes=7)  # One-hot encoding of labels
    #print(f'y_onehot: {y}, y_onehot.shape: {y.shape}')

    # DATA
    graph = data.Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)
    #print(graph)
    return graph


def addGraph(graph, subject):
    global total_data
    global tot


    total_data.append((graph, subject))
    tot += 1
    #print(f"\nTotal Data: {tot}")
    #print(total_data)


def create_dataloader(dataset, batch_size):
    # Create DataLoader with the specified dataset and batch size
    '''
     The DataLoader will group these Data objects into batches of the size specified by the user.
     This means that every time you iterate over trainLoader, you will get a batch of specified size of Data objects.
    '''

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #print(f"Created DataLoader for {dataset}")
    return loader


def save_dataloader(loader, filename):
    path = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\dataloaders\ " + filename + ".pt"
    torch.save(loader, path)
    print(f"Dataloader {filename} saved")

def load_dataloader(filename):
    path = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\dataloaders\ " + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader

def returnLocalizedFace(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    if len(faces) == 0:
        return img

    crop_img = img[y:y + h, x:x + w]
    return crop_img

def read(imagesPath, labels):
    # Function to read images from a given directory

    for filename in glob.glob(imagesPath + '/*.*'):
        print(filename.split(imagesPath + '/')[1])
        img = returnLocalizedFace(getImage(filename))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def getImage(path):
    return cv2.imread(path)

def show(img):
    cv2.imshow('im', img)
    cv2.waitKey(0)


subjectsTrain = []
subjectsTrain2 = []
subjectsTest = []
noneGraphs = 0
errorGraph = 0
distType = "manhattan"
numGraph = 0

dir_paths = [
        r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\ImageMORPHED_train\Cartella 1",
        r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\ImageMORPHED_train\Cartella 2",
    ]

def save_data_to_txt(data, file_path):
    with open(file_path, 'w' + '\n') as file:
        file.write(str(data))

def print_memory_usage():
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")

def print_memory_variable_usage (variable_name):
    memory_usage = sys.getsizeof(variable_name)
    print(f"Memory usage of {variable_name}: {memory_usage} bytes")

def image_generator_morphed(dir_path1, dist_type):
    for dir_path in dir_paths:
        print(f"\nProcessing images in folder: {dir_path}")
        if dir_path == r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\ImageBONAFIDE_test":
            for filename in os.listdir(dir_path):
                percorso_immagine = os.path.join(dir_path, filename)
                if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    num_graph = 0
                    foto_name = percorso_immagine.split(os.path.sep)[-1]
                    img = cv2.imread(percorso_immagine)
                    graph = mp.buildGraphNorm(img, dist_type)

                    if graph == 0:
                        error_graph = 1
                        print("file: ", filename)
                        print(
                            f"\n Graph with incorrect HOG feature: {num_graph} , number of graphs with errors: {error_graph}")
                        continue

                    if graph is None:
                        none_graphs = 1
                        print(f"\n Null graph index: {num_graph} , number of null graphs: {none_graphs}")
                        continue

                    subject = foto_name
                    if subject not in subjectsTrain:
                        subjectsTrain.append(subject)

                    label = 'morphed'
                    grafo = graph2Data(graph, label)

                    graph_data = graph2Data(grafo, label)

                    file_path = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\Windows\Build\grap2Data.txt"
                    # Chiamare la funzione per salvare il dato nel file
                    save_data_to_txt(graph_data, file_path)


                    # Stampiamo alcune informazioni sulla memoria
                    print_memory_usage()
                    print(f"Total Subjects: {len(subjectsTrain)}")

                    yield grafo, subject

'''
def image_generator_morphed(dir_path, dist_type):
    for filename in os.listdir(dir_path):
        percorso_immagine = os.path.join(dir_path, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            num_graph = 0
            foto_name = percorso_immagine.split(dir_path + '\\')[1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildGraphNorm(img, dist_type)

            if graph == 0:
                error_graph = 1
                print("file: ", filename)
                print(f"\n Graph with incorrect HOG feature: {num_graph} , number of graphs with errors: {error_graph}")
                continue

            if graph is None:
                none_graphs = 1
                print(f"\n Null graph index: {num_graph} , number of null graphs: {none_graphs}")
                continue

            subject = foto_name
            if subject not in subjectsTrain:
                subjectsTrain.append(subject)

            label = 'morphed'
            grafo = graph2Data(graph, label)
            yield grafo, subject
'''


def image_generator_bonafide(dir_path, dist_type):
    for filename in os.listdir(dir_path):
        percorso_immagine = os.path.join(dir_path, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            num_graph = 0
            foto_name = percorso_immagine.split(dir_path + '\\')[1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildGraphNorm(img, dist_type)

            if graph == 0:
                error_graph = 1
                print("file: ", filename)
                print(f"\n Graph with incorrect HOG feature: {num_graph} , number of graphs with errors: {error_graph}")
                continue

            if graph is None:
                none_graphs = 1
                print(f"\n Null graph index: {num_graph} , number of null graphs: {none_graphs}")
                continue

            subject = foto_name
            if subject not in subjectsTrain:
                subjectsTrain.append(subject)

            label = 'bonafide'
            grafo = graph2Data(graph, label)
            yield grafo, subject


#METODO PER LA REALIZZAZIONE DEL SET DI TRAINING E TESTING PER IMMAGINI MORPHATE
def beginLoopTrainTestSets():
    # Utilizzo del generatore
    distType = "manhattan"
    dirMorphedTrain = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\ImageBONAFIDE_test"
    dirMorphedTest = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\ImageMORPHED_test"
    dirBonafideTrain = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\ImageBONAFIDE_train"
    dirBonafideTest = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\ImageMORPHED_test"

    gen1 = image_generator_morphed(dirMorphedTrain, distType)
    #gen2 = image_generator_bonafide(dirBonafideTrain, distType)
    #gen3 = image_generator(dirMorphedTest, distType)
    #gen4 = image_generator(dirBonafideTest, distType)
    #print_memory_variable_usage(total_data)
    for grafo, subject in gen1:
        addGraph(grafo, subject)
    '''   
    for grafo, subject in gen2:
        addGraph(grafo, subject)
      
    print(len(subjectsTrain))  
    '''
    '''
    # Imposta il numero massimo di immagini da leggere prima di interrompere e riprendere
    max_images = 500
    numepoch = 10
    current_image_count = 0

    for epoch in range(0, numepoch):
        for filename in os.listdir(dirMorphedTrain):
            percorso_immagine = os.path.join(dirMorphedTrain, filename)

            if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                numGraph += 1
                current_image_count += 1
                fotoName = percorso_immagine.split(dirMorphedTrain + '\\')[1]
                img = cv2.imread(percorso_immagine)
                graph = mp.buildGraphNorm(img, distType)

                if graph == 0:
                    errorGraph += 1
                    print("file: ", filename)
                    print(f"\n Graph with incorrect HOG feature: {numGraph} , number of graphs with errors: {errorGraph}")
                    continue

                if graph is None:
                    noneGraphs += 1
                    print(f"\n Null graph index: {numGraph} , number of null graphs: {noneGraphs}")
                    continue

                subject = fotoName
                if subject not in subjectsTrain:
                    subjectsTrain.append(subject)

                label = 'morphed'
                grafo = graph2Data(graph, label)
                addGraph(grafo, subject)


                # Controlla se hai raggiunto il limite massimo di immagini da leggere
                if numGraph >= max_images:
                    print(current_image_count)
                    current_image_count = 0
                    break
    '''
    '''
    #FOR PER ITERARE LE IMMAGINI MORPHATE PER IL TRAIN
    for filename in os.listdir(dirMorphedTrain):
        percorso_immagine = os.path.join(dirMorphedTrain, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            numGraph += 1
            fotoName = percorso_immagine.split(dirMorphedTrain + '\\')[1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildGraphNorm(img, distType)

            if graph == 0:
                errorGraph += 1
                print("file: ", filename)
                print(f"\n Graph with incorrect HOG feature: {numGraph} , number of graphs with errors: {errorGraph}")
                continue

            if graph is None:
                noneGraphs += 1
                print(f"\n Null graph index: {numGraph} , number of null graphs: {noneGraphs}")
                continue

            subject = fotoName
            if subject not in subjectsTrain:
                subjectsTrain.append(subject)

            label = 'morphed'
            grafo = graph2Data(graph, label)
            addGraph(grafo, subject)
    '''
    '''
    #FOR PER ITERARE LE IMMAGINI MORPHATE PER IL TEST
    for filename in os.listdir(dirMorphedTest):
        percorso_immagine = os.path.join(dirMorphedTest, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            numGraph += 1
            fotoName = percorso_immagine.split(dirMorphedTest + '\\')[1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildGraph(img, distType)
            #mp.showGraph(img, distType)

            if graph == 0:
                errorGraph += 1
                print("file: ", filename)
                print(f"\n Graph with incorrect HOG feature: {numGraph} , number of graphs with errors: {errorGraph}")
                continue

            if graph is None:
                noneGraphs += 1
                print(f"\n Null graph index: {numGraph} , number of null graphs: {noneGraphs}")
                continue

            subject = fotoName
            if subject not in subjectsTest:
                subjectsTest.append(subject)

            label = "morphed"
            grafo = graph2Data(graph, label)
            addGraph(grafo, subject)
    '''
    '''
    # FOR PER ITERARE LE IMMAGINI BONAFIDE PER IL TRAIN
    for filename in os.listdir(dirBonafideTrain):
        percorso_immagine = os.path.join(dirBonafideTrain, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            numGraph += 1
            fotoName = percorso_immagine.split(dirBonafideTrain + '\\')[1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildGraphNorm(img, distType)

            if graph == 0:
                errorGraph += 1
                print("file: ", filename)
                print(
                    f"\n Graph with incorrect HOG feature: {numGraph} , number of graphs with errors: {errorGraph}")
                continue

            if graph is None:
                noneGraphs += 1
                print(f"\n Null graph index: {numGraph} , number of null graphs: {noneGraphs}")
                continue

            subject = fotoName
            if subject not in subjectsTrain:
                subjectsTrain.append(subject)

            label = 'bonafide'
            grafo = graph2Data(graph, label)
            addGraph(grafo, subject)
    '''
    '''
    #FOR PER ITERARE LE IMMAGINI BONAFIDE PER IL TEST
    for filename in os.listdir(dirBonafideTest):
        percorso_immagine = os.path.join(dirBonafideTest, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            numGraph += 1
            fotoName = percorso_immagine.split(dirBonafideTest + '\\')[1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildGraphNorm(img, distType)
            #mp.showGraph(img, distType)

            if graph == 0:
                errorGraph += 1
                print("file: ", filename)
                print(f"\n Graph with incorrect HOG feature: {numGraph} , number of graphs with errors: {errorGraph}")
                continue

            if graph is None:
                noneGraphs += 1
                print(f"\n Null graph index: {numGraph} , number of null graphs: {noneGraphs}")
                continue

            subject = fotoName
            if subject not in subjectsTest:
                subjectsTest.append(subject)

            label = "bonafide"
            grafo = graph2Data(graph, label)
            addGraph(grafo, subject)
    '''

    import random
    # Randomly shuffle the subjects
    random.shuffle(subjectsTrain)
    #random.shuffle(subjectsTest)

    print(f'Train subjects: {len(subjectsTrain)}')
    #print(f'Test subjects: {len(subjectsTest)}')



    # Filter the images by training and test subjects
    train_data = [graph for graph, subject in total_data if subject in subjectsTrain]
    #test_data = [graph for graph, subject in total_data if subject in subjectsTest]

    print(f'Train data: {len(train_data)}')
    # print(train_data)
    #print(f'Test data: {len(test_data)}')

    trainDataLoader = create_dataloader(train_data, 64)
    print("Number of batches in the train loader:", len(trainDataLoader))
    save_dataloader(trainDataLoader, "MBFTrainSMDD")
    #testDataLoader = create_dataloader(test_data, 64)
    #print("Number of batches in the test loader:", len(testDataLoader))
    #save_dataloader(testDataLoader, "MBFTestFRLL")

    '''
    for batch in trainDataLoader:
        for i in batch:
            for data_item, label in batch:
                print("Data item: ")
                print(data_item)
                print("Label: ")
                print(label)
                print("\n")
    '''


def test():
    noneGraphs = 0
    errorGraph = 0
    distType = "manhattan"
    subjects = []
    ok = 0
    numGraph = 0

    # Loop through the CK+ folders
    dir = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\Image_test"

    print(f'\nDirectory: {dir}')

    for filename in glob.glob(dir + '/*.*'):
        numGraph += 1
        fotoName = filename.split(dir + '\\')[1]

        img = cv2.imread(filename)

        # create the graph with HOG features
        graph = mp.buildGraphNorm(img, distType)

        if graph == 0:
            errorGraph += 1
            print("file: ", filename)
            print(f"\n Graph with incorrect HOG feature: {numGraph} , number of graphs with errors: {errorGraph}")
            continue

        if graph is None:
            noneGraphs += 1
            print(f"\n Null graph index: {numGraph} , number of null graphs: {noneGraphs}")
            continue

        subject = fotoName.split("_")[0]
        if subject not in subjects:
            subjects.append(subject)

        grafo = graph2DataWithHog(graph, 6, subject)
        addGraph(grafo, subject)
        ok += 1

    print(f"\nNumber of graphs with errors: {errorGraph}, Number of OK graphs: {ok} , ")



beginLoopTrainTestSets()
