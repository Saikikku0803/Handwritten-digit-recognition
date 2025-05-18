import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import numpy as np
from config import IMAGE_SIZE, NUM_CLASSES


def image_to_graph(image):
    h, w = IMAGE_SIZE
    edge_index = []
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i+dx, j+dy
                if 0 <= ni < h and 0 <= nj < w:
                    edge_index.append([idx, ni*w + nj])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    feat = torch.tensor(image / 255.0, dtype=torch.float).view(-1, 1)
    return feat, edge_index


class USPSGraphDataset(InMemoryDataset):
    def __init__(self, images, labels):
        super().__init__()
        data_list = []
        for img, label in zip(images, labels):
            x, edge_index = image_to_graph(img)
            y = torch.tensor([label], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.lin = torch.nn.Linear(128, NUM_CLASSES)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


def train_gcn_model(x_train, y_train, x_test, y_test, epochs=20):
    train_dataset = USPSGraphDataset(x_train, y_train)
    test_dataset = USPSGraphDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    all_pred = []
    for data in test_loader:
        out = model(data)
        pred = out.argmax(dim=1)
        all_pred.append(pred.detach().cpu())
        correct += (pred == data.y).sum().item()
    acc = correct / len(test_dataset)
    return model, torch.cat(all_pred), acc
