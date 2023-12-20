import pandas as pd
import cv2
import mediaPipeMeshModified
import numpy as np
import networkx as nx
import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transtorns
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset,Dataset


# Leggi il dataset originale FER2013
fer2013_df = pd.read_csv('fer2013.csv')
#print(f" lunghezza dataset {len(fer2013_df)}")

distType="manhattan"

# Crea una nuova lista di dizionari per i dati del nuovo dataset
train_data = []
validation_data = []
test_data = []

#quartaRiga=fer2013_df.iloc[5]

noneGraphs=0
train=0
test=0
val=0

from tqdm import tqdm


# Ciclo attraverso le prime 3 righe del dataset originale
for index in tqdm(range(1,5000), desc="Processing Rows"):
    row = fer2013_df.iloc[index]
    # Ottieni l'immagine dai pixel
    image_string = row['pixels'].split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    resized_image_data = cv2.resize(image_data, (64, 64), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(resized_image_data, cv2.COLOR_GRAY2BGR)  # Converti in formato BGR

    #print(f"img indice {index} : {image} ")
    #creazione del grafo
    graph = mediaPipeMeshModified.buildGraphWithGaborFeatures(image, distType)


    if(graph is None):
        noneGraphs+=1
        print(f"\n Grafo nullo indice: {index} , numero grafi nulli: {noneGraphs}")

        continue
    #print(f" grafo {index} {graph}")
    node_features = [list(graph.nodes[node]['gabor_features']) for node in graph.nodes]

    #tensore delle feature dei nodi
    x = torch.tensor(node_features, dtype=torch.float32)  # embeddings dei nodi o attributi dei nodi (gabor)

    # Mostra la forma del tensore x
    # print(x.shape)

    edges = list(graph.edges)
    #tensore degli archi
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # tensore degli archi del grafo
    # print(edge_index.shape)

    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    # print(edge_weights)

    # tensore attributo degli archi (distanza)
    edge_weights = torch.tensor(edge_weights, dtype=torch.long)  # tensore dei pesi/distanza degli archi
    # print(edge_weights.shape)

    #tensore label  target del grafo
    y = torch.tensor([row['emotion']], dtype=torch.long)  # etichetta emozione
    #print('y', y)

   #creazione oggetto Data Pytorch geom
    graph = data.Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=y)
    #Lista di oggetti Data suddivisi per utilizzo

    if row['Usage'] == 'Training':
        train_data.append(graph)
        train+=1
        print(f"\n Training: {train}, indice: {index}")

'''
#batch : collezione di Data objects
batch = data.Batch().from_data_list(train_data)
print("Number of graphs: ",batch.num_graphs)
print ("Graph at index I:",batch[1])
print ("Retrieve the list of graph\n: ",len(batch.to_data_list()))
'''
from torch_geometric.loader import DataListLoader

#dataset = DataListLoader(batch, batch_size=3, shuffle=False)

#Classe per creazione dataset di grafi
class  TrainDataset(InMemoryDataset):
    def __init__(self, root, transform=None):
        super(TrainDataset, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        #print('data', self.data, 'slices',self.slices )

    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['data.pt'] #file contenete dove salvare i dati

    def process(self):
        print('train process start')
        '''
        La funzione self.collate(graphList) prende una lista di oggetti Data (in questo caso graphList) 
        e li combina in un unico oggetto Data rappresentante il dataset completo. 
        Restituisce sia i dati aggregati data che le informazioni sulle porzioni slices. 
        '''
        data, slices = self.collate(train_data)
        torch.save((data, slices), self.processed_paths[0])



if train>0:
    train_dataset = TrainDataset("/Users/roby/PycharmProjects/FER/fer_train_data")
    length = len(train_dataset)
    print("Lunghezza del train dataset:", length)





