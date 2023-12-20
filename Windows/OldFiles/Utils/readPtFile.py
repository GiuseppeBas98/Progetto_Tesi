import torch
import torch_geometric.data as data
from torch_geometric.loader import DataLoader



test_loader = torch.load("D:\\PiscopoRoberto\\FER\\dataloaders\\trainTotalSubIndCk.pt")

print('\nTrain loader:')
for i, subgraph in enumerate(test_loader):
    print(f' - Subgraph {i}: {subgraph}')

for i,data in enumerate(test_loader):
    print(f' - Data {i}: {data}')
    print(f' - Data.x: {data.x}')
    print(f' - Data.y: {data.y}')
    print(f' - Data.edge_index: {data.edge_index}')
    print(f' - Data.batch: {data.batch}')
    print(f' - Data data.edge_weight: {data.edge_weight}')
'''
print()
#print(f'Dataset: {dataset}:')
print('====================')
num_elementi = len(dataset)
#print(f"Numero di elementi nel dataset: {num_elementi}")
print(f"Dataset: {dataset}")
print(f'Number of nodes: {dataset.num_nodes}')
print(f'Number of edges: {dataset[0].num_edges}')
print(f'Number of features: {dataset[0].num_features}')
print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}')
print(f'Has isolated nodes: {dataset[0].has_isolated_nodes()}')
print(f'Has self-loops: {dataset[0].has_self_loops()}')
print(f'Is undirected: {dataset[0].is_undirected()}')
print(f'Is directed: {dataset[0].is_directed()}')
'''

'''

batch = next(iter(dataset))
print("Batch:", batch)
print("Labels:", batch.y[:10])
print("Batch indices:", batch.batch[:40])
'''

'''
index=0
for batch in test_loader:
    index+=1
    print(f"Numero grafi nel batch {index}: ", batch.num_graphs)
    print(f"Graph at index {index}:", batch[index]) #oggetto Data
    data= batch[index]
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'Is directed: {data.is_directed()}')
'''
















