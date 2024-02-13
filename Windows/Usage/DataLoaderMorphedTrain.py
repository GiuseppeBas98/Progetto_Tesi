import cv2
import os
from Windows.Build import mediapipeMesh as mp
import torch
# from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
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


def addGraph(graph, subject):
    global total_data
    global tot

    total_data.append((graph, subject))
    tot += 1
    # print(f"\nTotal Data: {tot}")
    # print(total_data)


array = []
subjectsTrain = []
subjectsTest = []
noneGraphs = 0
distType = "manhattan"
num_grafico = 0


def print_memory_usage():
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")


def colleziona_grafo(dir_path):
    global array
    global num_grafico
    global noneGraphs
    global distType
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
            # if subject not in subjectsTrain:
            #     subjectsTrain.append(subject)

            label = 'morphed'
            grafo = graph2Data(graph, label)
            array.append(grafo)

            # addGraph(grafo, subject)
            # print_memory_usage()
            # print(f"Total Subjects: {len(subjectsTrain)}")
            # print(len(array))


def image_generator_morphed():
    with open("Cartella.txt", "r") as file:
        dir_path = file.read()
    print("Analizzo foto in Cartella: " + dir_path)
    colleziona_grafo(dir_path)


# METODO PER LA REALIZZAZIONE DEL SET DI TRAINING E TESTING PER IMMAGINI MORPHATE
def beginLoopTrain():
    image_generator_morphed()
    crea_dataLoader()


# METODI PER LA CREAZIONE DEL DATALOADER
def create_dataloader(dataset, batch_size):
    # Create DataLoader with the specified dataset and batch size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(f"Created DataLoader for {dataset}")
    return loader


def save_dataloader(loader, filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloaders/" + filename + ".pt"
    torch.save(loader, path)
    print(f"Dataloader {filename} saved")


def load_dataloader(filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloaders/" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader


def crea_dataLoader():
    global array
    path_dataloader_daEliminare = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\dataloaders\TrainDataloader_128Size.pt"

    # Verifica se il file esiste
    if os.path.exists(path_dataloader_daEliminare):
        # Se il file esiste, sovrascrivi il dataloader e salva
        data_loader = load_dataloader('TrainDataloader_128Size')
        dataset_originale = data_loader.dataset
        print("ARRAY VECCHIO: ")
        print(len(dataset_originale))

        array.extend(dataset_originale)
        print("ARRAY NUOVO: ")
        print(len(array))

        elimina_file_dataloader(path_dataloader_daEliminare)

        dataLoader = create_dataloader(array, batch_size=128)
        save_dataloader(dataLoader, 'TrainDataloader_128Size')
        dataset = dataLoader.dataset
        print("LUNGHEZZA: ")
        print(len(dataset))

    else:
        # Se il file non esiste, crea un nuovo dataloader e salva
        data_loader = create_dataloader(array, batch_size=128)
        save_dataloader(data_loader, 'TrainDataloader_128Size')
        # dataset = data_loader.dataset
        # print(dataset)


def elimina_file_dataloader(file_path):
    # Elimina il file associato
    try:
        os.remove(file_path)
        print(f"File eliminato: {file_path}")
    except OSError as e:
        print(f"Errore durante l'eliminazione del file: {e}")


def main():
    # print(torch.__version__)
    beginLoopTrain()
    # print("ARRAY: " + str(array))
    # d = load_dataloader('TrainDataloader')
    # # d1 = load_dataloader('TestDataloadermorph_opencv')
    # dset = d.dataset
    # # dset1 = d1.dataset
    # print("DSET:" + str(len(dset)))
    # # count = 0
    # print("STAMPO DATALOADER: ")
    # for data in d:
    #     print(data.y.long())
if __name__ == "__main__":
    main()