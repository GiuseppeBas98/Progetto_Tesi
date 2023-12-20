import os
import cv2
from otherFile import mediapipeMesh
import networkx as nx

distType="manhattan"

cartella = "/Users/roby/Uni/Tirocinio/FER/archive/test/angry"

# Ottieni una lista dei file nella cartella
immagini = os.listdir(cartella)

# Ordina la lista delle immagini in modo sequenziale
immagini.sort()

'''
for i, weight in enumerate(adjacency_matrix[274]):
    if weight != 0:
        node = list(g.nodes)[i]  # Ottieni il nodo corrispondente all'indice i
        print(f"Nodo {node}: peso {weight}")
'''

import csv

percorso_file_csv = "dataset.csv"  # Specifica il percorso del file CSV

dataset = []  # Lista per salvare i dati del dataset

i=0
count=2
# Itera per generare i dati per ogni immagine
for immagine in immagini:

    count = count - 1
    if(count>0):

        percorso_immagine = os.path.join(cartella, immagine)
        id_progressivo = i + 1
        image = cv2.imread(percorso_immagine)

        # ottieni la matrice di adiacenza del grafo
        g = mediapipeMesh.buildGraph(image, distType)

        adjacency_matrix = nx.to_numpy_array(g)
        #adjacency_matrix = adjacency_matrix.astype(int)




        # Aggiungi i dati all'elenco del dataset
        dataset.append({
            "Id": id_progressivo,
            "Adjacency_matrix": adjacency_matrix,
            "Category": "ANGRY"
    })

# Scrivi i dati nel file CSV
with open(percorso_file_csv, "w", newline="") as file_csv:
    writer = csv.DictWriter(file_csv, fieldnames=["Id", "Adjacency_matrix", "Category"])
    writer.writeheader()
    writer.writerows(dataset)





