import torch
from torch_geometric.loader import DataLoader


file_name = "graph2TrainDataMANHATTAN.txt"
file_path = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\Windows\Build\graph2TrainDataMANHATTAN.txt"


def create_dataloader(dataset, batch_size):
    # Create DataLoader with the specified dataset and batch size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(f"Created DataLoader for {dataset}")
    return loader


def save_dataloader(loader, filename):
    path = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\dataloaders\\" + filename + ".pt"
    torch.save(loader, path)
    print(f"Dataloader {filename} saved")


def load_dataloader(filename):
    path = r"C:\Users\Giuseppe Basile\Desktop\Tirocinio_Morphing_BG\dataloaders\\" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader

def load_data_from_txt(file_path):
    data = []
    c = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Ignora righe vuote
            parts = line.split('\t', 1)  # Limita il numero di divisioni a uno
            if len(parts) == 2:
                c += 1

                subject, grafo = parts
                data.append((subject, grafo))
            else:
                print(f"Warning: Ignorando la riga malformata: {line}")
    return data

def crea_dataLoader():
    global data_loader
    dataFromTxt = load_data_from_txt(file_path)
    data_loader = create_dataloader(dataFromTxt, batch_size=60)
    save_dataloader(data_loader, 'TrainDataloader')

    '''
    i = 0
    for batch in data_loader:
        # Estrai le tuple dai batch
        data, filename = batch
        i += 1
        print(i)
        print(data)
        print(filename)
    '''



crea_dataLoader()

