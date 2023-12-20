import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import CKPLUSGraphDataLoader as ck

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import numpy as np





class GCN(torch.nn.Module):
    """GCN"""

    def __init__(self, dim_h,num_node_features,num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h,num_classes)

    def forward(self, x, edge_index, batch, edge_weight):
        # Node embeddings
        h = self.conv1(x, edge_index,edge_weight)
        #if torch.isnan(h).any():
            #print("NaN values detected in h during training.")
        h = h.relu()
        h = self.conv2(h, edge_index,edge_weight)
        h = h.relu()
        h = self.conv3(h, edge_index,edge_weight)

        # Graph-level readout
        hG = global_add_pool(h, batch)
        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)

        return hG, F.log_softmax(h, dim=1)


class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_h,num_node_features,num_classes):
        super(GIN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.lin1 = Linear(dim_h * 3, dim_h * 3)
        self.lin2 = Linear(dim_h * 3, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)

def train(model, train_loader,test_loader,epochs,learning_rate,gnn,device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    accuracy_list=[]
    iteration_list=[]
    class_labels = ['Happy', 'Contempt','Fear','Disgust','Anger','Sadness','Surprise']
    fer_class_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    count = 0

    for epoch in range(epochs):
        total_loss = 0
        acc = 0
        index = 0

        model.train()
        # Train on batches
        for data in train_loader:
            data = data.to(device)  # Sposta i dati sul dispositivo corrente
            optimizer.zero_grad()
            index += 1
            if gnn == 'gcn':
                _, out = model(data.x, data.edge_index, data.batch, data.edge_weight)
            elif gnn == 'gin':
                _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.long())
            total_loss += loss / len(train_loader)
            acc += accuracy(out.argmax(dim=1), data.y.long()) / len(train_loader)
            loss.backward()
            optimizer.step()
            count += 1

            iteration_list.append(count)
            loss_list.append(loss.item())
            accuracy_list.append(acc)


        # Print metrics every 10 epochs
        if (epoch % 10 == 0):
            print(f'Epoch {epoch + 1:>3} | Train Loss: {total_loss:.2f} '
                  f'| Train Acc: {acc * 100:.2f}% ')

    plt.plot(iteration_list, loss_list)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.plot(iteration_list, accuracy_list,color="red")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy")
    plt.show()

    test_loss, test_acc,predicted_labels,true_labels = test(model, test_loader,gnn,device)

    fig, ax = plt.subplots(figsize=(11, 8))
    confusion = confusion_matrix(true_labels, predicted_labels)
    ConfusionMatrixDisplay(confusion, display_labels=fer_class_labels).plot(cmap='Blues', xticks_rotation="horizontal",ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')

    # Calcola precision, recall e F1-score
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')



    return model
@torch.no_grad()
def test(model, loader,gnn,device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    predicted_labels = []
    true_labels = []

    for data in loader:
        data = data.to(device)  # Sposta i dati sul dispositivo corrente
        if gnn == 'gcn':
            _, out = model(data.x, data.edge_index, data.batch, data.edge_weight)
        elif gnn == 'gin':
            _, out = model(data.x, data.edge_index, data.batch)
        #print(f"out in testMethod: {loss}")
        loss += criterion(out, data.y.long()) / len(loader)
        #print(f"loss in testMethod: {loss}, lunghezza loader: {len(loader)}")
        acc += accuracy(out.argmax(dim=1), data.y.long()) / len(loader)

        pred = out.argmax(dim=1)
        predicted_labels.extend(pred.tolist())
        true_labels.extend(data.y.long().tolist())

    return loss, acc,predicted_labels,true_labels

def accuracy(pred_y, y):
    """Calcola l'accuracy."""
    return ((pred_y == y).sum() / len(y)).item()




def start(gnn, epochs, learningRate, save, device):
    test_loader = ck.load_dataloader("testFERDataLoader")
    train_loader = ck.load_dataloader("trainFERDataLoader")

    if device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    if gnn=='gcn':
        model = GCN(dim_h=128,num_node_features=2,num_classes=7).to(device)
    elif gnn=='gin':
        model = GIN(dim_h=128,num_node_features=2,num_classes=7).to(device)

    model = train(model, train_loader,test_loader,epochs,learningRate,gnn,device)

    if save:
        torch.save(model.state_dict(), 'models/'+gnn+'_model.pth')

start('gcn',400,0.001,False,'gpu') #gcn',500,0.001  gin 400 0.0001