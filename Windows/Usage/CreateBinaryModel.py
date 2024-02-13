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


def train(model, train_loader, gnn, optimizer, criterion):
    total_loss = 0
    acc = 0
    index = 0
    global count, iteration_list, loss_list
    model.train()

    # Iterate in batches over the training dataset.
    for data in train_loader:
        index += 1
        count += 1
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
    return model, total_loss


def test(model, loader, gnn, criterion):
    model.eval()
    global accuracy_list
    plt.close("all")
    predicted_labels = []
    true_labels = []
    correct = 0
    loss = 0
    accuracy = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        if gnn == 'gcn':
            out = model(data.x, data.edge_index, data.batch, data.edge_weight)  # Perform a single forward pass.
        elif gnn == 'gin' or 'gcn_noedge':
            out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y.long()) / len(loader)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        predicted_labels.extend(pred.tolist())
        true_labels.extend(data.y.long().tolist())
        correct += int((pred == data.y.long()).sum())  # Check against ground-truth labels.
        accuracy = correct / len(loader.dataset)
        accuracy_list.append(accuracy)

    return accuracy, loss, predicted_labels, true_labels  # Derive ratio of correct predictions.


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
    plt.savefig('/Users/Giuseppe Basile/Desktop/New_Morphing/models/' + gnn + '_128SizeOpencv_Binary_model.png')
    plt.show()


def load_dataloader(filename):
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloaders/" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader


# start function used to begin the training phase
def start(gnn, epochs, learningRate, save):
    test_loader = load_dataloader("TestDataloadermorph_opencv")
    train_loader = load_dataloader("TrainDataloader_128Size")

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
        model, total_loss = train(model, train_loader, gnn, optimizer, criterion)
        train_acc, train_loss, _, _ = test(model, train_loader, gnn, criterion)
        test_acc, test_loss, predicted_labels, true_labels = test(model, test_loader, gnn, criterion)
        print(f'Epoch: {epoch:03d}, Train Loss: {total_loss:.2f},Train Acc: {train_acc:.4f}')
        # if (epoch % 10 == 0):
        #     print(f'Epoch: {epoch:03d}, Train Loss: {total_loss:.2f},Train Acc: {train_acc:.4f}')
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
                   '/Users/Giuseppe Basile/Desktop/New_Morphing/models/' + gnn + '_128SizeOpencv_Binary_model.pth')

# start('gcn', 6, 0.001, True)  # 150 epoche gin
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
#
# # Storing ID of current CUDA device
# cuda_id = torch.cuda.current_device()
# print(f"ID of current CUDA device:{torch.cuda.current_device()}")
#
# print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
