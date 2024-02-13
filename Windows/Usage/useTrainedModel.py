import torch
import CreateBinaryModel as cm
import numpy as np
from DataLoaderMorphedTrain import normalizeNodefeatures, normalizeValues
from Windows.Build import mediapipeMesh as mp
import cv2
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch_geometric.data as data
import os
from Windows.Utils import AlignImage


def graph2Data(graph):
    # Extract landmark coordinates from the graph
    landmark_coordinates = [graph.nodes[node]['pos'] for node in graph.nodes]
    landmark_coordinates_transposed = list(map(list, zip(*landmark_coordinates)))
    normalized_landmark_coordinates = list(map(list, zip(*normalizeNodefeatures(landmark_coordinates_transposed))))

    # Create a tensor 'x' with normalized coordinates
    x = torch.tensor(normalized_landmark_coordinates, dtype=torch.float32)

    # Extract edges from the graph
    edges = list(graph.edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Tensor of graph edges

    # Extract edge weights from the graph
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]

    # Normalize edge weights
    edge_weights_normalized = normalizeValues(edge_weights)

    # Create a tensor 'edge_weights' with normalized edge weights
    edge_weights = torch.tensor(edge_weights_normalized, dtype=torch.float32)

    # Create a PyTorch Data object
    graph_data = data.Data(x=x, edge_index=edge_index, edge_weight=edge_weights)

    return graph_data


def startClassification(gnn, image):
    classes = 2
    bin = '_Binary'
    class_labels = ['Morphed', 'Bonafide']

    if gnn == 'gcn':
        model = cm.GCN(dim_h=64, num_node_features=2, num_classes=classes)
    elif gnn == 'gin':
        model = cm.GIN(dim_h=64, num_node_features=2, num_classes=classes)

    # Path to the trained weights
    # path = "/Users/Giuseppe Basile/Desktop/New_Morphing/models/" + gnn + bin + "_model_.pth"
    path = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\models\gcn_CUDA60SizeOpencv_Binary_model.pth"
    print(path)

    # Load trained weights into the model
    model.load_state_dict(torch.load(path))
    model.eval()  # Evaluation mode
    distType = "manhattan"

    # Build the graph
    graph = mp.buildGraphNorm(image, distType)
    graph_data = graph2Data(graph)

    loader = DataLoader([graph_data], batch_size=1)

    # Make predictions
    with torch.no_grad():
        for data in loader:
            if gnn == 'gcn':
                predictions = model(data.x, data.edge_index, data.batch, data.edge_weight)
            elif gnn == 'gin':
                predictions, _ = model(data.x, data.edge_index, data.batch)

    # Extract the predicted class
    predicted_class = predictions.argmax(dim=1).item()

    # Label of the predicted class
    predicted_label = class_labels[predicted_class]

    # Position to write the label on your image
    position = (10, 30)

    # Text color and font size
    font_color = (0, 0, 255)  # Red
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Write the label on your image
    cv2.putText(image, predicted_label, position, font, font_scale, font_color, 2)

    # Display the image
    cv2.imshow('Image with Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# alignedFace = AlignImage.alignFace("/Users/Giuseppe Basile/Desktop/New_Morphing/imageTesting/img000001_B.png")
# image, gray_img = AlignImage.detectFace(alignedFace)
# plt.imshow(image[:, :, ::-1])
# plt.show()
# img = cv2.imread("/Users/Giuseppe Basile/Desktop/New_Morphing/imageTesting/img000001_B.png")
# startClassification('gcn', image)

dir_path = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\imageTesting"
for filename in os.listdir(dir_path):
    percorso_immagine = os.path.join(dir_path, filename)
    # alignedFace = AlignImage.alignFace(percorso_immagine)
    # image, gray_img = AlignImage.detectFace(alignedFace)
    image = cv2.imread(percorso_immagine)
    startClassification('gcn', image)
