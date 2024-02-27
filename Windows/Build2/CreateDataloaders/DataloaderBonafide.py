import cv2
import os
from Windows.Build import mediapipeMesh as mp
import torch
import random
import torch_geometric.data as data
import psutil
from torch_geometric.loader import DataLoader
from tqdm import tqdm

train = 0
test = 0
val = 0
tot = 0

train_data = []
validation_data = []
test_data = []
total_data = []
array = []
subjects = []
train_subjects = []
test_subjects = []
val_subjects = []
noneGraphs = 0
distType = "manhattan"
num_grafico = 0
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


def graph2Data(graph, type):
    # NODES
    # Extract landmark coordinates
    landmark_coordinates = [graph.nodes[node]['pos'] for node in graph.nodes]

    # Transpose the list of coordinates to have a list of coordinates per dimension
    landmark_coordinates_transposed = list(map(list, zip(*landmark_coordinates)))

    # Transpose normalized coordinates
    normalized_landmark_coordinates = list(map(list, zip(*normalizeNodefeatures(landmark_coordinates_transposed))))

    x = torch.tensor(normalized_landmark_coordinates, dtype=torch.float32)  # Tensor x with normalized coordinates
    # print(f'Shape of tensor x: {x.shape}, Node Features Tensor: {x}')

    # EDGES
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Tensor of graph edges
    # print(edge_index.shape)
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    edge_weights_normalized = normalizeValues(edge_weights)  # Normalize values
    edge_weights = torch.tensor(edge_weights_normalized, dtype=torch.float32)  # Tensor of edge weights/distances
    # print(edge_weights.shape)
    # print('label', emotion)
    # LABEL
    if type == 'morphed':
        y = 0
    elif type == 'bonafide':
        y = 1
    y = torch.tensor(y, dtype=torch.long)
    # print(y)
    # print(f'y: {y}')
    # y = F.one_hot(y, num_classes=7)  # One-hot encoding of labels
    # print(f'y_onehot: {y}, y_onehot.shape: {y.shape}')

    # DATA
    graph = data.Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)
    # print(graph)
    # print(graph.x, graph.y, graph.edge_weight, graph.edge_index)
    return graph

def print_memory_usage():
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")


def colleziona_grafo(dir_path):
    global array, subjects, train_subjects, test_subjects, val_subjects
    global num_grafico, noneGraphs, distType
    file_list = os.listdir(dir_path)
    for filename in tqdm(file_list, desc="Elaborazione Immagini"):
        percorso_immagine = os.path.join(dir_path, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            num_grafico += 1
            foto_name = percorso_immagine.split(os.path.sep)[-1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildGraphNorm(img, distType)

            if graph is None:
                noneGraphs += 1
                print(f"\n Null graph index: {num_grafico} , number of null graphs: {noneGraphs}")
                continue

            subject = foto_name
            if subject not in subjects:
                subjects.append(subject)

            label = 'bonafide'
            grafo = graph2Data(graph, label)
            array.append((grafo, subject))

    # Randomly shuffle the subjects
    random.shuffle(subjects)

    # Suddivisione del 70% per addestramento, 15% per validazione, 15% per test
    train_subjects = subjects[:int(0.7 * len(subjects))]
    val_subjects = subjects[int(0.7 * len(subjects)):int(0.85 * len(subjects))]
    test_subjects = subjects[int(0.85 * len(subjects)):]

    # Filter the images by training, validation, and test subjects
    train_data = [graph for graph, subject in array if subject in train_subjects]
    val_data = [graph for graph, subject in array if subject in val_subjects]
    test_data = [graph for graph, subject in array if subject in test_subjects]

    crea_dataLoader(train_data, val_data, test_data)

def image_generator_bonafide():
    with open("CartellaMerged.txt", "r") as file:
        dir_path = file.read()
    print("Analizzo foto in Cartella: " + dir_path)
    colleziona_grafo(dir_path)


# METODO PER LA REALIZZAZIONE DEL SET DI TRAINING E TESTING PER IMMAGINI MORPHATE
def beginLoopTrain():
    image_generator_bonafide()


# METODI PER LA CREAZIONE DEL DATALOADER
def create_dataloader(dataset, batch_size):
    # Create DataLoader with the specified dataset and batch size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(f"Created DataLoader for {dataset}")
    return loader


def save_dataloader(loader, filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloadersMerged/" + filename + ".pt"
    torch.save(loader, path)
    print(f"Dataloader {filename} saved")


def load_dataloader(filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloadersMerged/" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader


def crea_dataLoader(datasetTrain, datasetVal, datasetTest):
    global array
    path_dataloaderTrain_daEliminare = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\dataloadersMerged\TrainDataloader_128Size.pt"
    path_dataloaderTest_daEliminare = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\dataloadersMerged\TestDataloader_128Size.pt"
    path_dataloaderValidation_daEliminare = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\dataloadersMerged\ValidationDataloader_128Size.pt"

    # Verifica se il file esiste
    if os.path.exists(path_dataloaderTrain_daEliminare) and os.path.exists(path_dataloaderTest_daEliminare) and os.path.exists(path_dataloaderValidation_daEliminare):
        # Se il file esiste, sovrascrivi il dataloader e salva
        data_loader_train = load_dataloader('TrainDataloader_128Size')
        data_loader_test = load_dataloader('TestDataloader_128Size')
        data_loader_val = load_dataloader('ValidationDataloader_128Size')
        dataset_originale_train = data_loader_train.dataset
        dataset_originale_test = data_loader_test.dataset
        dataset_originale_val = data_loader_val.dataset

        datasetTrain.extend(dataset_originale_train)
        datasetTest.extend(dataset_originale_test)
        datasetVal.extend(dataset_originale_val)

        elimina_file_dataloader(path_dataloaderTrain_daEliminare)
        elimina_file_dataloader(path_dataloaderTest_daEliminare)
        elimina_file_dataloader(path_dataloaderValidation_daEliminare)

        new_dataLoader_Train = create_dataloader(datasetTrain, batch_size=128)
        new_dataLoader_Test = create_dataloader(datasetTest, batch_size=128)
        new_dataLoader_Val = create_dataloader(datasetVal, batch_size=128)
        save_dataloader(new_dataLoader_Train, 'TrainDataloader_128Size')
        save_dataloader(new_dataLoader_Test, 'TestDataloader_128Size')
        save_dataloader(new_dataLoader_Val, 'ValidationDataloader_128Size')
        dataset1 = new_dataLoader_Train.dataset
        dataset2 = new_dataLoader_Test.dataset
        dataset3 = new_dataLoader_Val.dataset
        print('Len di train:' + str(len(dataset1)) + '\n' + 'Len di test:' + str(len(dataset2)) + '\n' + 'Len di val:' + str(len(dataset3)))

    else:
        # Se il file non esiste, crea un nuovo dataloader e salva
        data_loader_train = create_dataloader(datasetTrain, batch_size=128)
        data_loader_test = create_dataloader(datasetTest, batch_size=128)
        data_loader_val = create_dataloader(datasetVal, batch_size=128)
        save_dataloader(data_loader_train, 'TrainDataloader_128Size')
        save_dataloader(data_loader_test, 'TestDataloader_128Size')
        save_dataloader(data_loader_val, 'ValidationDataloader_128Size')
        dataset_train = data_loader_train.dataset
        dataset_test = data_loader_test.dataset
        dataset_val = data_loader_val.dataset
        print('len di train:' + str(len(dataset_train)) + '\n' + 'len di test:' + str(len(dataset_test)) + '\n' + 'len di val:' + str(len(dataset_val)))


def elimina_file_dataloader(file_path):
    # Elimina il file associato
    try:
        os.remove(file_path)
        print(f"File eliminato: {file_path}")
    except OSError as e:
        print(f"Errore durante l'eliminazione del file: {e}")


def main():
    # print(torch.__version__)
    # beginLoopTrain()
    # print("ARRAY: " + str(array))
    d = load_dataloader('TrainDataloader_128Size')
    d1 = load_dataloader('TestDataloader_128Size')
    d2 = load_dataloader('ValidationDataloader_128Size')
    dset = d.dataset
    dset1 = d1.dataset
    dset2 = d2.dataset
    print('len di train:' + str(len(dset)) + '\n' + 'len di test:' + str(
        len(dset1)) + '\n' + 'len di val:' + str(len(dset2)))
    # count = 0
    # c = 0
    # print("STAMPO DATALOADER: ")
    # for data in dset:
    #     if data.y == 0:
    #         count += 1
    #     if data.y == 1:
    #         c += 1
    # print(count)
    # print(c)


if __name__ == "__main__":
    main()