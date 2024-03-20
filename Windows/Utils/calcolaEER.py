import torch
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from Windows.Usage.CreateBinaryModelWITHCUDA import GCN, GAT, GIN, test


def load_dataloader(filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloaders/" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader


# Esempio di utilizzo
# Sostituisci con il tuo modello e dataloader
model = GCN(dim_h=64, num_node_features=2, num_classes=2)
# model = GIN(dim_h=64, num_node_features=2, num_classes=2)
# model = GAT(dim_h=128, num_node_features=2, num_classes=2, num_heads=2)
model_path = r'C:\Users\Giuseppe Basile\Desktop\New_Morphing\models\normalModels\gcn_CUDA128SizeAmslFacemorpher_Binary_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
dataloader = load_dataloader('TestDataloadermorph_amsl')
criterion = torch.nn.CrossEntropyLoss()
test_acc, test_loss, predicted_labels, true_labels = test(model, dataloader, 'gcn', criterion)
fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
print(f'Equal Error Rate (EER): {eer}')
