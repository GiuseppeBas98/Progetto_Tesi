import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical

'''
data = pd.read_csv("fer2013.csv")
data.head()

print("Numero di righe e colonne di dati= ", data.shape)
print("Nome delle colonne = ", data.columns)

data["Usage"].value_counts()

training = data.loc[data["Usage"] == "Training"]
public_test = data.loc[data["Usage"] == "PublicTest"]
private_test = data.loc[data["Usage"] == "PrivateTest"]

print("Training set ", training.shape)
print("Public test = ", public_test.shape)
print("Private test =", private_test.shape)

print("========================= Emotion Counter ===========================")
print("train test= \n{}, \npublic test = \n{}, \nprivate test = \n{}".format(training["emotion"].value_counts(),
      public_test["emotion"].value_counts(), private_test["emotion"].value_counts()))


train_labels = training["emotion"]
train_labels = to_categorical(train_labels)

train_pixels = training["pixels"].str.split(" ").tolist()
train_pixels = np.uint8(train_pixels)
train_pixels = train_pixels.reshape((28709, 48, 48, 1))
train_pixels = train_pixels.astype("float32") / 255


private_labels = private_test["emotion"]
private_labels = to_categorical(private_labels)

private_pixels = private_test["pixels"].str.split(" ").tolist()
private_pixels = np.uint8(private_pixels)
private_pixels = private_pixels.reshape((3589, 48, 48, 1))
private_pixels = private_pixels.astype("float32") / 255


public_labels = public_test["emotion"]
public_labels = to_categorical(public_labels)

public_pixels = public_test["pixels"].str.split(" ").tolist()
public_pixels = np.uint8(public_pixels)
public_pixels = public_pixels.reshape((3589, 48, 48, 1))
public_pixels = public_pixels.astype("float32") / 255


plt.figure(0, figsize=(12,6))
for i in range(1, 13):
    plt.subplot(3,4,i)
    plt.imshow(train_pixels[i, :, :, 0], cmap="gray")


plt.tight_layout()
plt.show()
'''

import pandas as pd
import numpy as np
from PIL import Image
import mediaPipeMeshModified
import cv2
import networkx as nx

df = pd.read_csv('fer2013.csv')
for image_pixels in df.iloc[1:,1]: #column 2 has the pixels. Row 1 is column name.
    image_string = image_pixels.split(' ') #pixels are separated by spaces.
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48)
    resized_image_data = cv2.resize(image_data, (64, 64), interpolation=cv2.INTER_LINEAR)
    #img = Image.fromarray(image_data) #final image
    break
distType="manhattan"
image = cv2.cvtColor(resized_image_data, cv2.COLOR_GRAY2BGR)


graph, gabor_features = mediaPipeMeshModified.buildGraph(image, distType)
for i, node in enumerate(graph.nodes()):
    graph.nodes[node]['features'] = gabor_features[i]

# Itera attraverso i nodi del grafo e stampa i vettori delle feature associate
for node in graph.nodes():
    features = graph.nodes[node]['features']
    num_features = len(features)
    print(f"Nodo {node}: Numero di feature = {num_features}, Feature = {features}")




#disegno nodi grafo
#nx.draw(graph,with_labels=True)
#plt.show()
# matrice di adiacenza del grafo
a=nx.to_scipy_sparse_array(graph)
print(a)

