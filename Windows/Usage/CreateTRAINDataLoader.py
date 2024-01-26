import torch
from torch_geometric.loader import DataLoader
import torch_geometric.data as data

file_name = "graph2TrainDataMANHATTAN.txt"
file_path = r"C:\Users\Giuseppe Basile\Desktop\New_Morphing\Windows\Build\prova.txt"


class Datatorcia:
    def __init__(self, soggetto, x, edge_weight, edge_index, y):
        self.soggetto = soggetto
        self.x = x
        self.edge_weight = edge_weight
        self.edge_index = edge_index
        self.y = y

    def get_soggetto(self):
        return self.soggetto

    def set_soggetto(self, new_sog):
        self.x = new_sog

    def get_x(self):
        return self.x

    def set_x(self, new_x):
        self.x = new_x

    def get_edge_weight(self):
        return self.edge_weight

    def set_edge_weight(self, new_edge_weight):
        self.edge_weight = new_edge_weight

    def get_edge_index(self):
        return self.edge_index

    def set_edge_index(self, new_edge_index):
        self.edge_index = new_edge_index

    def get_y(self):
        return self.y

    def set_y(self, new_y):
        self.y = new_y

    def __str__(self):
        return f"Data(x={self.x}, edge_weight={self.edge_weight}, edge_index={self.edge_index}, y={self.y})"



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
    path = "/Users/Giuseppe Basile/Desktop/New_Morphing/dataloaders/" + filename + ".pt"
    loader = torch.load(path)
    print(f"Dataloader {filename} loaded")
    return loader

def load_data_from_txt(file_path):
    d = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Rimuovi il carattere di nuova linea alla fine e dividi la riga
            data = line.strip().split('\t')
            # Estrai i valori dalla lista data e crea un oggetto Dati
            subject = data[0].split(':')[1]
            print(subject)
            x = torch.tensor(data[1].split(':')[1])
            print(x)
            edgeW = float(data[2].split(':')[1])
            edgeI = int(data[3].split(':')[1])
            y = float(data[4].split(':')[1])
            temp = Datatorcia(subject, x, edgeW, edgeI, y)
            d.append(temp)
    return d

def crea_dataLoader():
    global data_loader

    dataFromTxt = load_data_from_txt(file_path)
    data_loader = create_dataloader(dataFromTxt, batch_size=60)
    save_dataloader(data_loader, 'TrainDataloaderProva')


    i = 0
    for batch in data_loader:
        # Estrai le tuple dai batch
        print(batch)






crea_dataLoader()

