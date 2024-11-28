# %%
import sys
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
from dataset.data import *

# %%

def gnn_data_obj(data_list):
        
        # Takes a list of graph files and returns a list of PyG Data objects

        data_obj_list = []
        
        for file in tqdm(data_list):
         
            G = nx.read_gml(file)
        
            # STEP 1: CREATE FEATURE TENSORS FOR THE NODES

            x = []

            for node in G.nodes:
                x.append(list(G.nodes[node]['history']))
            
            scaler = MinMaxScaler()

            x = scaler.fit_transform(x)
            
            x = torch.tensor(x, dtype=torch.float)

            # STEP 2: CREATE EDGE TENSORS FOR THE EDGES

            edge_index = []

            edge_attr = []

            for edge in G.edges:
                edge_index.append([list(G.nodes).index(edge[0]), list(G.nodes).index(edge[1])])
                edge_attr.append(G.edges[edge]['weight'])

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            # STEP 3: CREATE TARGET TENSORS FOR THE NODES

            y = []

            for node in G.nodes:
                y.append(G.nodes[node]['target'])

            y = torch.tensor(y, dtype=torch.float)

            # Unsqueeze the target tensor

            y = y.unsqueeze(1)

            # Store the graph in a PyG Data object
        
            data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y)

            data_obj_list.append(data)

        return data_obj_list

class StockGCN(torch.nn.Module):

    # A 3-layer GCN for node regression
    
    def __init__(self):
        
        super(StockGCN, self).__init__()

        self.conv1 = GCNConv(30, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 1)
    
    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        
        return x
    
class StockGAT(torch.nn.Module):

    # A 3-layer GAT for node regression
    
    def __init__(self):
        
        super(StockGAT, self).__init__()

        self.conv1 = GATConv(30, 64, heads = 4, concat = True, dropout = 0.5)
        self.conv2 = GATConv(64*4, 64, heads = 4, concat = True, dropout = 0.5)
        self.conv3 = GATConv(64*4, 1, heads = 4, concat = False, dropout = 0.5)        
    
    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.5, training=self.training)
        
        x = self.conv3(x, edge_index)

        return x


# Training StockGNN

def train(model, optimizer, train_loader, test_loader, num_epochs):

    model.train()
    average_train_loss = []
    average_test_mape = []
    
    for epoch in tqdm(range(num_epochs)):
        
        total_train_loss = 0
        
        for data in train_loader:

            # data.to(device)                
            optimizer.zero_grad()

            out = model(data)

            # Compute the Node Regression Loss

            loss = F.l1_loss(out, data.y) # Mean Absolute Error

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_loss = total_train_loss/len(train_loader)
        average_train_loss.append(average_loss)
        test_mape = eval(model, test_loader)
        average_test_mape.append(test_mape)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {average_loss:.4f}, Val MAPE: {test_mape:.4f}')

    return average_train_loss, average_test_mape


def eval(model, test_loader):

    model.eval()
    total_mape = 0
    
    with torch.no_grad():

        for data in test_loader:

            out = model(data)

            # Compute the Mean Absolute Percentage Error

            mape = torch.mean(torch.abs((data.y - out) / (data.y + 1e-7)))

            total_mape += mape.item()
            
    return total_mape/len(test_loader)