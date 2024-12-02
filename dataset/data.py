# # %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")

# print("Path to dataset files:", path)

"""
SERVES THE FOLLOWING PURPOSES:

1) Reading and cleaning the data: filter and only include data from the last 6 years for a SUBSET of stocks
compute daily return, remove outliers, and normalize the data

2) For every stock, based on the daily returns, compute pairwise correlations.

3) Creating a graph object where node features: ADJUSTED closing values for last 30 days
edge weights: correlation between the stocks + binary variable to indicate if a pair of nodes belong
to the same stock sector (optional).

"""

import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import networkx as nx
from datetime import datetime
from models.baselines import *

# %%
class NASDAQDataset():

    def __init__(self, stocks_path, meta_path, nasdaq_100_path, start_date):
        
        # Read all the csv files in stocks_path and store their names in a list
        
        self.stock_path = stocks_path
        self.nasdaq_100 = pd.read_csv(nasdaq_100_path)['SYMBOL'].tolist() # List of stocks in the NASDAQ 100 index
        self.start_date = start_date
        self.raw_stock_list = [os.path.basename(path).split(".")[0] for path in glob.glob(f'{self.stock_path}/*.csv')]
        self.meta_data = pd.read_csv(meta_path)
        self.data = {}
        self.filtered_stock_list = []

    # Read the data for each stock and store it in a dictionary

    def read_data(self):

        for stock in tqdm(self.nasdaq_100): # Only considering stocks in the NASDAQ 100 index                              
            
            if stock in self.raw_stock_list:
                
                data = pd.read_csv(f'{self.stock_path}/{stock}.csv')
                data['Date'] = pd.to_datetime(data['Date'])
                data = data[data['Date'] >= self.start_date]

                if len(data) >= 1500: # Set this number to get a higher number of data points (stocks will lesser data points are filtered out)
                    self.data[stock] = data
                    self.data[stock] = data.set_index('Date')

            else:
                
                print(f"Data for {stock} is missing!")


        # Find the intersection of dates for all stocks in self.data
        
        date_sets = {stock: set(self.data[stock].index) for stock in self.data.keys()}
        common_dates = set.intersection(*date_sets.values())

        # Filter stock data to only include common dates
        for stock in self.data.keys():
            self.data[stock] = self.data[stock].loc[sorted(common_dates)] # Takes care of gaps in data

        # Compute daily returns for each stock

        for stock in self.data.keys():
            self.data[stock]['return'] = self.data[stock]['Adj Close'].pct_change() # Daily returns: computes percentage change in closing price
            self.data[stock] = self.data[stock].dropna(subset=['return'])

        self.filtered_stock_list = list(self.data.keys())

    def compute_correlation_matrix(self, start_date):

        # Computes the correlation matrix for all stocks between start date and (start date - 30) days

        return_df = pd.DataFrame()
        
        for stock in self.filtered_stock_list:            
                
            start_idx = self.data[stock].index.get_loc(start_date)
            return_df[stock] = self.data[stock]['return'].iloc[max(0, start_idx - 30):start_idx] # Get return values for last 30 days

        # Compute the correlations matrix

        correlation_matrix = return_df.corr()

        return correlation_matrix, return_df

    
    def daily_graph_generator(self, start_date):
        
        # Generates a graph for each day and stores it as a GML file

        data_dir = '/Users/vivek/Documents/PhD/UIUC/Fall24/CS598/Project/GNN-Stock-Market-Prediction/dataset/graphs'
        
        # Convert start_date to string

        start_date = start_date.strftime('%Y-%m-%d')
        filename = f'{data_dir}/{start_date}.gml'

        if not os.path.exists(filename):
            
            G = nx.Graph()
            G.add_nodes_from(self.filtered_stock_list) 
            correlation_matrix, _ = self.compute_correlation_matrix(start_date)
            threshold = 0.98
            
            for stock in G.nodes:
                
                start_idx = self.data[stock].index.get_loc(start_date)
                G.nodes[stock]['history'] = self.data[stock]['Adj Close'].iloc[max(0, start_idx + 1 - 30):start_idx + 1].tolist() # Store closing values for last 30 days
                G.nodes[stock]['target'] = self.data[stock]['Adj Close'].iloc[start_idx + 1] # Store the closing value for the next day        
                G.nodes[stock]['linr_regr'] = float(linear_regression(np.arange(30).reshape(-1, 1), G.nodes[stock]['history']))
                G.nodes[stock]['mov_avg'] = float(moving_average(G.nodes[stock]['history'], 5))
                G.nodes[stock]['exp_smoothing'] = float(simple_exp_smoothing(G.nodes[stock]['history'], 0.3))
                G.nodes[stock]['holt_winters'] = float(holt_winters(G.nodes[stock]['history']))
                                  

                
            for i in range(len(self.filtered_stock_list)):
                for j in range(i, len(self.filtered_stock_list)):
                    if abs(correlation_matrix.iloc[i, j]) >= threshold: # Correlations can negative/positive; hence the absolute value
                        G.add_edge(self.filtered_stock_list[i], self.filtered_stock_list[j], weight = abs(correlation_matrix.iloc[i, j]))
        
            nx.write_gml(G, filename)

            return G

        else:
            
            print(f"Graph for {start_date} already exists!")
            return nx.read_gml(filename)

