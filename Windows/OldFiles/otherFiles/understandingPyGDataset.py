from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='.', name='PROTEINS').shuffle()

# Print information about the dataset
print(f'Dataset: {dataset}')
print('-------------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {dataset[0].x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

test_dataset  = dataset[int(len(dataset)*0.9):]
print(f'Test set       = {len(test_dataset)} graphs')

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print('\nTest loader:')
for i, subgraph in enumerate(test_loader):
    print(f' - Subgraph {i}: {subgraph}')

for i,data in enumerate(test_loader):
    print(f' - Data {i}: {data}')
    print(f' - Data.x: {data.x}')
    print(f' - Data.edge_index: {data.edge_index}')
    print(f' - Data.batch: {data.batch}')

for data in test_dataset:
    assert data.edge_index.max() < data.num_nodes
    print(f' - data.edge_index.max(): {data.edge_index.max()} ,data.num_nodes: {data.num_nodes}')

