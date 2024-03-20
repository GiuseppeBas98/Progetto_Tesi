import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

loss_list = []
accuracy_list = []
iteration_list = []
count = 0



class GAT(torch.nn.Module):
    """GAT"""

    def __init__(self, dim_h, num_node_features, num_classes, num_heads=1):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_node_features, dim_h, heads=num_heads)
        self.conv2 = GATConv(dim_h * num_heads, dim_h, heads=num_heads)
        self.conv3 = GATConv(dim_h * num_heads, dim_h, heads=num_heads)
        self.lin = Linear(dim_h * num_heads, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # Graph-level readout
        x = global_add_pool(x, batch)

        # Classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN_NOEDGE(torch.nn.Module):
    """GCN"""

    def __init__(self, dim_h, num_node_features, num_classes):
        super(GCN_NOEDGE, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Graph-level readout
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)

        # Classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN(torch.nn.Module):
    """GCN"""

    def __init__(self, dim_h, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, num_classes)

    def forward(self, x, edge_index, batch, edge_weight):
        # Node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # Graph-level readout
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)

        # Classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self, dim_h, num_node_features, num_classes):
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
        self.lin1 = Linear(dim_h, dim_h)
        self.lin2 = Linear(dim_h, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        # Graph-level readout
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)

        # Classifier
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


def train(model, train_loader, val_loader, gnn, optimizer, criterion):
    global count, iteration_list, loss_list
    total_loss = 0
    val_loss = 0
    acc = 0
    index = 0

    model.train()

    # Muovi il modello sulla GPU se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for data in train_loader:
        index += 1
        count += 1

        data = data.to(device)  # Muovi i dati sulla GPU se disponibile

        if gnn == 'gcn':
            out = model(data.x, data.edge_index, data.batch, data.edge_weight)
        elif gnn == 'gin' or gnn == 'gcn_noedge' or gnn == 'gat':
            out = model(data.x, data.edge_index, data.batch)

        loss = criterion(out, data.y.long())
        total_loss += loss / len(train_loader)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iteration_list.append(count)
        loss_list.append(loss.item())

    val_loss = validate(model, val_loader, criterion, gnn)

    return model, total_loss, val_loss


def validate(model, val_loader, criterion, gnn):
    model.eval()
    val_loss = 0

    # Muovi il modello sulla GPU se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for val_data in val_loader:
            val_data = val_data.to(device)  # Muovi i dati sulla GPU se disponibile
            if gnn == 'gcn':
                val_out = model(val_data.x, val_data.edge_index, val_data.batch, val_data.edge_weight)
            elif gnn == 'gin' or gnn == 'gcn_noedge' or gnn == 'gat':
                val_out = model(val_data.x, val_data.edge_index, val_data.batch)

            val_loss += criterion(val_out, val_data.y.long()) / len(val_loader)

    model.train()

    return val_loss


def test(model, loader, gnn, criterion):
    model.eval()
    global accuracy_list
    plt.close("all")
    predicted_labels = []
    true_labels = []
    correct = 0
    loss = 0
    accuracy = 0

    # Muovi il modello sulla GPU se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with torch.no_grad():
        for data in loader:  # Iterate in batches over the test dataset.
            data = data.to(device)  # Muovi i dati sulla GPU se disponibile

            if gnn == 'gcn':
                out = model(data.x, data.edge_index, data.batch, data.edge_weight)
            elif gnn == 'gin' or gnn == 'gcn_noedge' or gnn == 'gat':
                out = model(data.x, data.edge_index, data.batch)

            loss += criterion(out, data.y.long()).item() / len(loader)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            predicted_labels.extend(pred.cpu().tolist())  # Sposta le predizioni sulla CPU per l'estensione della lista
            true_labels.extend(data.y.long().cpu().tolist())  # Sposta le etichette reali sulla CPU
            correct += int((pred == data.y.long()).sum())  # Check against ground-truth labels.

        accuracy = correct / len(loader.dataset)
        accuracy_list.append(accuracy)

    return accuracy, loss, predicted_labels, true_labels


def buildAndShowConfusionMatrix(true_labels, predicted_labels, gnn):
    global name_final_file
    class_labels = ["Morphed", "Bonafide"]
    # change font dict inside confusion matrix
    number_font_dict = {
        'weight': 'bold',
        'size': 40,
    }
    label_font_dict = {
        'weight': 'bold',
        'size': 45,
        'color': 'black'
    }
    class_label_font_dict = {
        'weight': 'bold',
        'size': 40,
        'color': 'black'
    }
    title_font_dict = {
        'family': 'sans-serif',
        'variant': 'normal',
        'weight': 'bold',
        'size': 60,
        'color': 'black'
    }
    fig, ax = plt.subplots(figsize=(30, 20))
    confusion = confusion_matrix(true_labels, predicted_labels)
    ConfusionMatrixDisplay(confusion, display_labels=class_labels).plot(
        cmap='Blues',
        xticks_rotation="horizontal",
        ax=ax,
        text_kw=number_font_dict,
        colorbar=False
    )
    # Personalize the x and y font size
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=class_label_font_dict)
    ax.set_yticklabels(ax.get_yticklabels(), fontdict=class_label_font_dict)

    # Draw vertical lines between columns
    for i in range(confusion.shape[1] - 1):
        ax.axvline(x=i + 0.5, color='black', linewidth=1)

    # Draw horizontal lines between rows
    for i in range(confusion.shape[0] - 1):
        ax.axhline(y=i + 0.5, color='black', linewidth=1)

    plt.xlabel('Predicted Label', fontdict=label_font_dict)
    plt.ylabel('True Label', fontdict=label_font_dict)
    plt.title(gnn, fontdict=title_font_dict)

    # Save the plt graphic as image
    plt.savefig(
        '/Users/Giuseppe Basile/Desktop/New_Morphing/models3/' + gnn + name_final_file + '_Binary_model.png')
    plt.show()


def load_dataloader(filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloaders/" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader


name_final_file = '_CUDA128FacemorpherAmsl'


# start function used to begin the training phase
def start(gnn, epochs, learningRate, save, patience):
    global name_final_file
    name_train_d, name_vale_d, name_test_d = 'TrainDataloader_128Size', 'TestDataloadermorph_amsl', 'TestDataloadermorph_facemorpher'
    train_loader = load_dataloader(name_train_d)
    val_loader = load_dataloader(name_vale_d)
    test_loader = load_dataloader(name_test_d)

    train_acc_list, val_acc_list, test_acc_list, epochs_list, train_loss_list, val_loss_list = [], [], [], [], [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # checks on the type of model
    if gnn == 'gcn':
        model = GCN(dim_h=128, num_node_features=2, num_classes=2).to(device)
    elif gnn == 'gin':
        model = GIN(dim_h=128, num_node_features=2, num_classes=2).to(device)
    elif gnn == 'gcn_noedge':
        model = GCN_NOEDGE(dim_h=128, num_node_features=2, num_classes=2).to(device)
    elif gnn == 'gat':
        model = GAT(dim_h=128, num_node_features=2, num_classes=2, num_heads=2).to(device)

    weight_decay = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)  # lr=0.0001
    criterion = torch.nn.CrossEntropyLoss()

    # Crea un DataFrame per memorizzare i dati
    df = pd.DataFrame(
        columns=['Train Set', 'Val Set', 'Test Set', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc',
                 'Test Loss', 'Test Acc', 'Precision',
                 'Recall', 'F1-Score', 'EER', 'Comments'])

    temp_df = pd.DataFrame({'Train Set': [name_train_d],
                            'Val Set': [name_vale_d],
                            'Test Set': [name_test_d],
                            })

    # Concatena il DataFrame temporaneo al DataFrame principale
    df = pd.concat([df, temp_df], ignore_index=True)

    for epoch in range(1, epochs):
        model, total_loss, val_loss = train(model, train_loader, val_loader, gnn, optimizer, criterion)
        train_acc, train_loss, _, _ = test(model, train_loader, gnn, criterion)
        val_acc, val_loss, _, _ = test(model, val_loader, gnn, criterion)  # Valutazione sul set di validazione
        test_acc, test_loss, predicted_labels, true_labels = test(model, test_loader, gnn, criterion)

        # Append accuracy and loss values to the lists
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        # test_acc_list.append(test_acc)
        epochs_list.append(epoch)

        # Crea un DataFrame temporaneo per i nuovi dati
        temp_df = pd.DataFrame({'Epoch': [epoch],
                                'Train Loss': ['{:.3f}'.format(total_loss)],
                                'Train Acc': ['{:.3f}'.format(train_acc)],
                                'Val Loss': ['{:.3f}'.format(val_loss)],
                                'Val Acc': ['{:.3f}'.format(val_acc)],
                                'Test Loss': ['{:.3f}'.format(test_loss)],
                                'Test Acc': ['{:.3f}'.format(test_acc)]})

        # Concatena il DataFrame temporaneo al DataFrame principale
        df = pd.concat([df, temp_df], ignore_index=True)

        if (epoch % 10 == 0):
            print(
                f'Epoch: {epoch:03d}, Train Loss: {total_loss:.3f}, Train Acc: {train_acc:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}')

        # Controlla se la perdita di validazione ha migliorato
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # Verifica se interrompere l'addestramento
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break

    # Get precision, recall and F1-score
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Get ROC curve and calculate EER
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # Crea un DataFrame temporaneo per i dati di precision, recall e f1-score
    temp_df = pd.DataFrame({'Precision': ['{:.3f}'.format(precision)],
                            'Recall': ['{:.3f}'.format(recall)],
                            'F1-Score': ['{:.3f}'.format(f1)],
                            'EER': ['{:.3f}'.format(eer)],
                            'Comments': [
                                'lr = 0.00001\noptimizer = Adam\npatience = 1500\nglobal_mean_pool\ndim_h = 128\nnum_node_features=1']})

    # Concatena il DataFrame temporaneo dei dati finali al DataFrame principale
    df = pd.concat([df, temp_df], ignore_index=True)

    # Salva il DataFrame formattato in un file CSV
    csv_file_path = '/Users/Giuseppe Basile/Desktop/New_Morphing/models3/' + f'{gnn}{name_final_file}_Details.csv'
    df.to_csv(csv_file_path, index=False, float_format='%.2f')

    # GRAPH FOR ACCURACY
    # Plot the accuracy graphs
    plt.plot(epochs_list, train_acc_list, label='Train Accuracy')
    plt.plot(epochs_list, val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the accuracy graph in the same directory as the model
    save_dir = '/Users/Giuseppe Basile/Desktop/New_Morphing/models3/'
    plt.savefig(os.path.join(save_dir, f'{gnn}{name_final_file}_Accuracy_Graph.png'))
    plt.show()

    # GRAPH FOR LOSS
    # Plot the loss graphs
    plt.plot(epochs_list, train_loss_list, label='Train Loss')
    plt.plot(epochs_list, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the loss graph in the same directory as the model
    save_dir = '/Users/Giuseppe Basile/Desktop/New_Morphing/models3/'
    plt.savefig(os.path.join(save_dir, f'{gnn}{name_final_file}_Loss_Graph.png'))
    plt.show()

    # CONFUSION MATRIX
    # Plotting
    buildAndShowConfusionMatrix(true_labels, predicted_labels, gnn)

    # Print and save precision, recall, and F1-score
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}')
    print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}, EER: {eer:.3f}')

    if save == True:
        torch.save(model.state_dict(),
                   '/Users/Giuseppe Basile/Desktop/New_Morphing/models3/' + gnn + name_final_file + '_Binary_model.pth')


def main():
    start('gat', 1000, 0.00001, True, 100)  # 150 epoche gin


if __name__ == "__main__":
    main()
