import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import numpy as np

loss_list = []
accuracy_list = []
iteration_list = []
count = 0


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
        x = global_add_pool(x, batch)
        # x= global_mean_pool(x,batch)

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
        x = global_add_pool(x, batch)
        # x= global_mean_pool(x,batch)

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
        x = global_add_pool(x, batch)

        # Classifier
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


# def train(model, train_loader, gnn, optimizer, criterion):
#     total_loss = 0
#     acc = 0
#     index = 0
#     global count, iteration_list, loss_list
#     model.train()
#
#     # Sposta il modello sulla GPU se CUDA è disponibile
#     if torch.cuda.is_available():
#         model = model.to('cuda')
#
#     # Iterate in batches over the training dataset.
#     for data in train_loader:
#         index += 1
#         count += 1
#
#         # Sposta i dati sulla GPU se CUDA è disponibile
#         if torch.cuda.is_available():
#             data.x, data.edge_index, data.batch, data.edge_weight, data.y = \
#                 data.x.to('cuda'), data.edge_index.to('cuda'), data.batch.to('cuda'), data.edge_weight.to(
#                     'cuda'), data.y.to('cuda')
#
#         if gnn == 'gcn':
#             out = model(data.x, data.edge_index, data.batch, data.edge_weight)
#         elif gnn == 'gin' or gnn == 'gcn_noedge':
#             out = model(data.x, data.edge_index, data.batch)
#
#         loss = criterion(out, data.y.long())
#         total_loss += loss / len(train_loader)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         iteration_list.append(count)
#         loss_list.append(loss.item())
#
#     return model, total_loss

def train(model, train_loader, val_loader, gnn, optimizer, criterion):
    total_loss = 0
    acc = 0
    index = 0
    global count, iteration_list, loss_list
    model.train()

    # Sposta il modello sulla GPU se CUDA è disponibile
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Iterate in batches over the training dataset.
    for data in train_loader:
        index += 1
        count += 1

        # Sposta i dati sulla GPU se CUDA è disponibile
        if torch.cuda.is_available():
            data.x, data.edge_index, data.batch, data.edge_weight, data.y = \
                data.x.to('cuda'), data.edge_index.to('cuda'), data.batch.to('cuda'), data.edge_weight.to(
                    'cuda'), data.y.to('cuda')

        if gnn == 'gcn':
            out = model(data.x, data.edge_index, data.batch, data.edge_weight)  # Perform a single forward pass.
        elif gnn == 'gin' or 'gcn_noedge':
            out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.long())  # Compute the loss.
        total_loss += loss / len(train_loader)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients

        iteration_list.append(count)
        loss_list.append(loss.item())  # Use item() to get the scalar value of the loss

    # Validation loss without parameter update
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            if torch.cuda.is_available():
                val_data.x, val_data.edge_index, val_data.batch, val_data.edge_weight, val_data.y = \
                    val_data.x.to('cuda'), val_data.edge_index.to('cuda'), val_data.batch.to(
                        'cuda'), val_data.edge_weight.to(
                        'cuda'), val_data.y.to('cuda')

            if gnn == 'gcn':
                val_out = model(val_data.x, val_data.edge_index, val_data.batch, val_data.edge_weight)
            elif gnn == 'gin' or 'gcn_noedge':
                val_out = model(val_data.x, val_data.edge_index, val_data.batch)
            val_loss += criterion(val_out, val_data.y.long()) / len(val_loader)

    model.train()

    return model, total_loss, val_loss


def test(model, loader, gnn, criterion):
    model.eval()
    global accuracy_list
    plt.close("all")
    predicted_labels = []
    true_labels = []
    correct = 0
    loss = 0
    accuracy = 0

    # Sposta il modello sulla GPU se CUDA è disponibile
    if torch.cuda.is_available():
        model = model.to('cuda')

    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            # Sposta i dati sulla GPU se CUDA è disponibile
            if torch.cuda.is_available():
                data.x, data.edge_index, data.batch, data.edge_weight, data.y = \
                    data.x.to('cuda'), data.edge_index.to('cuda'), data.batch.to('cuda'), data.edge_weight.to(
                        'cuda'), data.y.to('cuda')

            if gnn == 'gcn':
                out = model(data.x, data.edge_index, data.batch, data.edge_weight)
            elif gnn == 'gin' or gnn == 'gcn_noedge':
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
    plt.title('GCN', fontdict=title_font_dict)

    # Save the plt graphic as image
    plt.savefig('/Users/Giuseppe Basile/Desktop/New_Morphing/models/' + gnn + '_CUDA60SizeOpencv_Binary_model.png')
    plt.show()


def load_dataloader(filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloaders/" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader


# start function used to begin the training phase
def start(gnn, epochs, learningRate, save):
    test_loader = load_dataloader("TestDataloadermorph_facemorpher")
    train_loader = load_dataloader("TrainDataloader")
    val_loader = load_dataloader('TestDataloadermorph_amsl')

    # checks on the type of model
    if gnn == 'gcn':
        model = GCN(dim_h=64, num_node_features=2, num_classes=2)
    elif gnn == 'gin':
        model = GIN(dim_h=64, num_node_features=2, num_classes=2)
    elif gnn == 'gcn_noedge':
        model = GCN_NOEDGE(dim_h=64, num_node_features=2, num_classes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)  # lr=0.0001
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs):
        model, total_loss, val_loss = train(model, train_loader, val_loader, gnn, optimizer, criterion)
        train_acc, train_loss, _, _ = test(model, train_loader, gnn, criterion)
        val_acc, val_loss, _, _ = test(model, val_loader, gnn, criterion)  # Valutazione sul set di validazione
        test_acc, test_loss, predicted_labels, true_labels = test(model, test_loader, gnn, criterion)
        # print(f'Epoch: {epoch:03d}, Train Loss: {total_loss:.2f},Train Acc: {train_acc:.4f}')
        if (epoch % 10 == 0):
            print(f'Epoch: {epoch:03d}, Train Loss: {total_loss:.2f},Train Acc: {train_acc:.4f},Val Acc: {val_acc:.4f}')
    # plot confusion matrix
    buildAndShowConfusionMatrix(true_labels, predicted_labels, gnn)

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    # Get precision, recall and F1-score
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    # save the trained model
    if save == True:
        torch.save(model.state_dict(),
                   '/Users/Giuseppe Basile/Desktop/New_Morphing/models/' + gnn + '_CUDA60SizeOpencv_Binary_model.pth')


#
# # Verifica se CUDA è disponibile
# if torch.cuda.is_available():
#     # Stampa le informazioni sulla versione di CUDA
#     print(f"Versione di CUDA disponibile: {torch.version.cuda}")
#
#     # Stampa le informazioni sulla GPU
#     for i in range(torch.cuda.device_count()):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("CUDA non è disponibile. Verifica l'installazione di CUDA e PyTorch con CUDA.")

def main():
    # print(torch.__version__)
    start('gcn', 20, 0.001, True)  # 150 epoche gin


if __name__ == "__main__":
    main()
