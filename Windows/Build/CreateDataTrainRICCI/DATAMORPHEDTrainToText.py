import cv2
import os
from Windows.Build import mediapipeMesh as mp
import torch
import torch_geometric.data as data
from tqdm import tqdm
# import tensorflow as tf
from torch_geometric.loader import DataLoader
import glob
import psutil
import sys
import csv

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
    # print(f'y: {y}, y_shape: {y.shape}')
    # y = F.one_hot(y, num_classes=7)  # One-hot encoding of labels
    # print(f'y_onehot: {y}, y_onehot.shape: {y.shape}')

    # DATA
    graph = data.Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)
    # print(graph)
    return graph


def addGraph(graph, subject):
    global total_data
    global tot

    total_data.append((graph, subject))
    tot += 1
    # print(f"\nTotal Data: {tot}")
    # print(total_data)



subjectsTrain = []
subjectsTest = []
noneGraphs = 0
distType = "manhattan"
num_grafico = 0
file_name = "../graph2DataMANHATTAN.txt"
file_path = r"/Windows/Build/graph2DataMANHATTAN.txt"



def print_memory_usage():
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")


def save_to_txt(subject, grafo):
    # Scrivi la riga nel file txt
    with open(file_name, 'a') as file:
        line = f"{grafo}\t{subject}\n"
        file.write(line)


def colleziona_grafo(dir_path):
    global num_grafico
    global distType
    for filename in os.listdir(dir_path):
        percorso_immagine = os.path.join(dir_path, filename)
        if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            num_graph = 0
            foto_name = percorso_immagine.split(os.path.sep)[-1]
            img = cv2.imread(percorso_immagine)
            graph = mp.buildOllivierRicciGraph(img, distType)

            if graph is None:
                none_graphs = 1
                print(f"\n Null graph index: {num_graph} , number of null graphs: {none_graphs}")
                continue

            subject = foto_name
            if subject not in subjectsTrain:
                subjectsTrain.append(subject)

            label = 'morphed'
            grafo = graph2Data(graph, label)

            # Chiamare la funzione per salvare il dato nel file
            save_to_txt(subject, grafo)

            # Stampiamo alcune informazioni sulla memoria
            print_memory_usage()
            print(f"Total Subjects: {len(subjectsTrain)}")

def image_generator_morphed():
    #for dir_path in dir_paths:
        #print(f"\nProcessing images in folder: {dir_path}")
        #if dir_path == r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\datasets\SMDD_dataset\m15k_t\SubFolder_Morphed_1":
    with open("Cartella.txt", "r") as file:
        dir_path = file.read()
    print("Analizzo foto in Cartella: " + dir_path)
    colleziona_grafo(dir_path)


# METODO PER LA REALIZZAZIONE DEL SET DI TRAINING E TESTING PER IMMAGINI MORPHATE
def beginLoopTrain():
    image_generator_morphed()


beginLoopTrain()

