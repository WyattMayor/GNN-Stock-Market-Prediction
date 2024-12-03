# %%
import sys
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
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
            
            # scaler = MinMaxScaler()
            # x = scaler.fit_transform(x)
            # x = torch.tensor(x, dtype=torch.float)

            # Convert to NumPy array for normalization
            
            # x = np.array(x)

            # # Z-score normalization per node
            # mean = x.mean(axis=1, keepdims=True)  # Mean over time for each node
            # std = x.std(axis=1, keepdims=True)    # Std over time for each node
            # x = (x - mean) / (std + 1e-9)         # Avoid division by zero
            
            # # Convert back to PyTorch tensor
            
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
            linr_regr = []
            mov_avg = []
            exp_smoothing = []
            holt_winters = []


            for node in G.nodes:
                
                y.append(G.nodes[node]['target'])
                linr_regr.append(G.nodes[node]['linr_regr'])
                mov_avg.append(G.nodes[node]['mov_avg'])
                exp_smoothing.append(G.nodes[node]['exp_smoothing'])
                holt_winters.append(G.nodes[node]['holt_winters'])


            y = torch.tensor(y, dtype=torch.float)
            linr_regr = torch.tensor(linr_regr, dtype=torch.float)
            mov_avg = torch.tensor(mov_avg, dtype=torch.float)
            exp_smoothing = torch.tensor(exp_smoothing, dtype=torch.float)
            holt_winters = torch.tensor(holt_winters, dtype=torch.float)

            # Unsqueeze the target tensor

            y = y.unsqueeze(1)
            linr_regr = linr_regr.unsqueeze(1)
            mov_avg = mov_avg.unsqueeze(1)
            exp_smoothing = exp_smoothing.unsqueeze(1)
            holt_winters = holt_winters.unsqueeze(1)            

            # Store the graph in a PyG Data object
        
            data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y, linr_regr = linr_regr, mov_avg = mov_avg, exp_smoothing = exp_smoothing, holt_winters = holt_winters)

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

        self.conv1 = GATConv(30, 128, heads = 4, concat= False, dropout = 0.3)
        self.conv2 = GATConv(128*4, 128, heads = 4, concat = False, dropout = 0.3)
        self.conv3 = GATConv(128*4, 1, heads = 4, concat = False, dropout = 0.3)        
    
    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p = 0.3, training=self.training)
        
        x = self.conv3(x, edge_index)

        return x

class StockGraphSAGE(torch.nn.Module):
    # A 3-layer GraphSAGE for node regression
    
    def __init__(self):
        super(StockGraphSAGE, self).__init__()
        
        self.conv1 = SAGEConv(30, 128, aggr = 'mean')
        self.conv2 = SAGEConv(128, 256, aggr = 'mean')
        self.conv3 = SAGEConv(256, 1, aggr = 'mean')
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.to('cuda')
        edge_index = edge_index.to('cuda')
        edge_attr = edge_attr.to('cuda')
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv3(x, edge_index) # To ensure output is compatible with the target shape
        
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
            data = data.to('cuda')
            out = model(data)
            out = out.to('cuda')
            # Compute the Node Regression Loss

            # loss = F.l1_loss(out, data.y) # Mean Absolute Error

            loss = F.mse_loss(out, data.y) # Mean Squared Error

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
            out = out.to('cuda')
            # Compute the Mean Absolute Percentage Error
            data = data.to('cuda')
            mape = torch.mean(torch.abs((data.y - out) / (data.y + 1e-7)))

            total_mape += mape.item()
            
    return total_mape/len(test_loader)
# %%

