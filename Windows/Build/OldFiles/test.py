import glob

from Windows.Build import mediapipeMesh as mp
import cv2
import torch
import torch_geometric.data as data


def normalizeNodefeatures(coordinates):
    # Inizializza le liste per le coordinate normalizzate
    normalized_landmark_coordinates = []

    # Normalizza ciascuna dimensione separatamente
    for dim_coordinates in coordinates:
        min_val = min(dim_coordinates)
        max_val = max(dim_coordinates)
        normalized_dim_coordinates = [(x - min_val) / (max_val - min_val) for x in dim_coordinates]
        normalized_landmark_coordinates.append(normalized_dim_coordinates)

    return normalized_landmark_coordinates

def normalizeValues(list):
    # Calcola il valore minimo e massimo
    minValue = min(list)
    maxValue = max(list)

    # Normalizza i valori dutilizzando la formula Min-Max
    list_normalized = [(el - minValue) / (maxValue - minValue) for el in list]
    return list_normalized




def graph2Data(graph,emotion,sub=None):

    landmark_coordinates = [graph.nodes[node]['pos'] for node in graph.nodes]

    hog_features = [graph.nodes[node]['hog'] for node in graph.nodes]

    #print("len hog",len(hog_features))
    landmark_coordinates_transposed = list(map(list, zip(*landmark_coordinates)))
    normalized_landmark_coordinates = list(map(list, zip(*normalizeNodefeatures(landmark_coordinates_transposed))))
    final_features = []
    for _ in range(36):
        final_features.append([])
    for i in range(36):
        #print("Lunghezza normalized_landmark_coordinates[{}]: {}".format(i, len(normalized_landmark_coordinates[i])))
        #print("Lunghezza hog_features[{}]: {}".format(i, len(hog_features[i])))

        for coord in normalized_landmark_coordinates[i]:
            final_features[i].append(coord)
        for hog in hog_features[i]:
            final_features[i].append(hog)

        #print("Lunghezza finale di final_features[{}]: {}".format(i, len(final_features[i])))


    #for sublist in final_features:
        #print(len(sublist))
    x= torch.tensor(final_features, dtype=torch.float32)
    #print(x)
    #EDGES
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    edge_weights_normalized = normalizeValues(edge_weights)
    edge_weights = torch.tensor(edge_weights_normalized, dtype=torch.float32)

    #LABEL
    y = torch.tensor(emotion, dtype=torch.float)

   # sub_tensor = torch.tensor([ord(char) for char in sub])
    #DATA
    graph = data.Data(x=x, edge_index=edge_index,edge_weight=edge_weights, y=y,subject=sub)

    return graph

#import bozza
distType="manhattan"







from skimage.feature import hog
from skimage import exposure
def compute_hog_1(image):
    """
    Calcola l'istogramma delle orientazioni (HOG) per un'immagine.
    Restituisce il vettore delle feature HOG.
    """

    # Calcola l'HOG dell'immagine
    features, _= hog(image, orientations=9, pixels_per_cell = (8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
    #for i, f in enumerate(features):
        #print(f' features {i}: {features[i]}')
    # Normalizza l'istogramma
    features = exposure.rescale_intensity(features, in_range=(0, 10))


    return features

def compute_hog_for_node_1(image, node_position, block_size=8):
    """
    Calcola l'HOG per un nodo in una regione centrata attorno al nodo.
    """
    # Estrai la regione intorno al nodo
    x,y=pos
    region = image[y - block_size:y + block_size, x - block_size:x + block_size]
    nuova_dimensione = (48, 48)
    # Usa il metodo cv2.resize per ridimensionare l'immagine
    immagine_ridimensionata = cv2.resize(region, nuova_dimensione)
    cv2.imshow("img",immagine_ridimensionata)
    cv2.waitKey(0)
    # Calcola l'HOG per la regione
    hog_features = compute_hog_1(region)

    return hog_features




'''
def compute_hog(image):
    """
    Calcola l'istogramma delle orientazioni (HOG) per un'immagine.
    Restituisce il vettore delle feature HOG.
    """
    win_size = (12, 12)
    block_size = (12, 12)
    block_stride = (6, 6)
    cell_size = (12, 12)
    nbins = 9

    # Crea un oggetto HOGDescriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    # Calcola l'HOG dell'immagine
    features = hog.compute(image)

    # Normalizza l'istogramma
    features = features.flatten()

    return features

def compute_hog_for_node(image, node_position, block_size=12):
    """
    Calcola l'HOG per un nodo in una regione centrata attorno al nodo.
    """
    x, y = node_position

    # Estrai la regione intorno al nodo
    region = image[y - block_size:y + block_size, x - block_size:x + block_size]
    cv2.imshow("img", region)
    cv2.waitKey(0)
    # Calcola l'HOG per la regione
    hog_features = compute_hog(region)

    print(len(hog_features))
    for i, f in enumerate(hog_features):
        print(f' features {i}: {hog_features[i]}')

    return hog_features

'''

def graph2Data(graph, emotion):
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
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Tensor of graph edges
    #print(edge_index.shape)
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    edge_weights_normalized = normalizeValues(edge_weights)  # Normalize values
    edge_weights = torch.tensor(edge_weights_normalized, dtype=torch.float32)  # Tensor of edge weights/distances
    #print(edge_weights.shape)
    # print('label', emotion)
    # LABEL
    y = torch.tensor(emotion, dtype=torch.float)  # Tensor label, target emotion for the graph
    #print(f'y: {y}, y_shape: {y.shape}')
    #y = F.one_hot(y, num_classes=7)  # One-hot encoding of labels
    #print(f'y_onehot: {y}, y_onehot.shape: {y.shape}')

    # DATA
    graph = data.Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)

    return graph


pos=(105, 193)

import os

'''
dir = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\Image_test"
numGraph = 0
errorGraph = 0
noneGraphs = 0
subjectsTrain = []

# Percorso della cartella contenente le immagini

# Cicla attraverso i file nella cartella
for filename in os.listdir(dir):
    percorso_immagine = os.path.join(dir, filename)
    if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(percorso_immagine)
        numGraph += 1
        img = cv2.imread(percorso_immagine)
        graph = mp.buildGraph(img, distType)
        mp.showGraph(img,distType)

#image = cv2.imread(r"C:\Users\Giuseppe Basile\Desktop\thesis-L31-main\Windows\FER\morphed_img000184_img282038.png")
#graph = mp.buildGraph(image, distType)


#mp.showGraph(image,distType)
#graph2Data(graph,2)

#image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#print(len(compute_hog_for_node_1(image_gray,pos)))
'''
'''

num_epochs = 5
    images_per_epoch = 1000
    controllo = 0
    for epoch in range(num_epochs):
        # FOR PER ITERARE LE IMMAGINI PER IL TRAIN
        for filename in os.listdir(dirTrain):
            percorso_immagine = os.path.join(dirTrain, filename)
            if os.path.isfile(percorso_immagine) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                numGraph += 1
                controllo += 1
                fotoName = percorso_immagine.split(dirTrain + '\\')[1]
                img = cv2.imread(percorso_immagine)
                graph = mp.buildGraph(img, distType)

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

                label = 'morphed'
                grafo = graph2Data(graph, label)
                addGraph(grafo, subject)
            # Controllo se abbiamo raggiunto il numero desiderato di immagini per epoca
            if controllo >= images_per_epoch:
                print(epoch)
                print(numGraph)
                controllo = 0
                break

'''