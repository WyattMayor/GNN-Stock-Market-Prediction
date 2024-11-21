# %%
import sys
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv
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

class StockGNN(torch.nn.Module):

    # A 2-layer GCN with an MLP head for node regression
    
    def __init__(self):
        super(StockGNN, self).__init__()

        self.conv1 = GCNConv(30, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)
        self.lin1 = torch.nn.Linear(128, 1)
    
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
        
        x = self.lin1(x)       
        
        return x
    

# Training StockGNN

def train_gnn(model, optimizer, train_loader, test_loader, num_epochs):

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
        val_mape = eval_gcn(model, test_loader)
        average_test_mape.append(val_mape)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {average_loss:.4f}, Val MAPE: {val_mape:.4f}')

    return average_train_loss, average_test_mape


def eval_gcn(model, test_loader):

    model.eval()
    total_mape = 0
    
    with torch.no_grad():

        for data in test_loader:

            out = model(data)

            # Compute the Mean Absolute Percentage Error

            mape = torch.mean(torch.abs(out - data.y)/data.y).item()

            total_mape += mape

    return total_mape/len(test_loader)
              



# %%
data_path = '/Users/vivek/Documents/PhD/UIUC/Fall24/CS598/Project/GNN-Stock-Market-Prediction/dataset/graphs'
data_list = [os.path.join(data_path, file) for file in os.listdir(data_path)]
# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stock_model = StockGNN().to(device)
optimizer = torch.optim.Adam(stock_model.parameters(), lr=0.005, weight_decay=1e-5)


# %%
train_loader = gnn_data_obj(data_list[0:1000])
test_loader = gnn_data_obj(data_list[1000:])

# %%

train_loss, test_loss = train_gnn(stock_model, optimizer, train_loader, test_loader, 300)

# %%

# Plot the training and validation loss

# plt.plot(train_loss, label = 'Training Loss')
plt.plot(test_loss, label = 'Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Percentage Error')
plt.show()

# %%

pred = stock_model(train_loader[0])
label = test_loader[0].y

abs_perc_error = torch.mean(100 * (torch.abs(pred - label)/label))