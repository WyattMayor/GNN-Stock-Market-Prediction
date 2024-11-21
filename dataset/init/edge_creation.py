import pandas as pd
import numpy as np
import json
from typing import List, Optional
import networkx as nx


def compute_daily_correlation(stock1_data: pd.DataFrame, stock2_data: pd.DataFrame, date: str) -> Optional[float]:
    """
    Computes the correlation between two stocks on a specific day using 'Open', 'High', 'Low', and 'Close' values.

    Args:
        stock1_data (pd.DataFrame): DataFrame for stock 1 with 'Date', 'Open', 'High', 'Low', and 'Close' columns.
        stock2_data (pd.DataFrame): DataFrame for stock 2 with 'Date', 'Open', 'High', 'Low', and 'Close' columns.
        date (str): The date for which to compute the correlation.

    Returns:
        Optional[float]: Correlation for the specific day or None if data is insufficient.
    """
    #Filter data for the specified date
    stock1_data = stock1_data[stock1_data['Date'] == pd.to_datetime(date)]
    stock2_data = stock2_data[stock2_data['Date'] == pd.to_datetime(date)]
    
    #Ensure we have data for both stocks on the specified date
    if stock1_data.empty or stock2_data.empty:
        return None

    #Extract 'Open', 'High', 'Low', 'Close' values from the last day for each stock
    features1 = stock1_data[['Open', 'High', 'Low', 'Close']].values.flatten()
    features2 = stock2_data[['Open', 'High', 'Low', 'Close']].values.flatten()

    #Compute correlation between the feature sets of the two stocks
    correlation = np.corrcoef(features1, features2)[0, 1]

    return correlation


def create_daily_graph(dataset, stock_list, date, days_lookback = 30):
    """
    Generates a graph for a specific date with stocks as nodes and their correlation as edge weights.

    Args:
        dataset (NASDAQDataset): Dataset instance containing loaded stock data.
        stock_list (List[str]): List of stock tickers to include.
        date (str): The specific date for which to create the graph.
        days_lookback (int): Number of days back to use as node features.

    Returns:
        G (networkx.Graph): Graph of stocks with correlations as edge weights.
    """
    #Create graph
    G = nx.Graph()
    
    #Add nodes with historical closing prices as features
    for stock in stock_list:
        stock_data = dataset.get_ticker_data(stock)
        stock_data = stock_data[stock_data['Date'] <= date].tail(days_lookback)  # Last `days_lookback` days
        G.add_node(stock, features=stock_data['Close'].values.tolist())
    
    #Add edges with correlation based on the final closing day
    for i, stock1 in enumerate(stock_list):
        stock1_data = dataset.get_ticker_data(stock1)
        
        for j in range(i + 1, len(stock_list)):
            stock2 = stock_list[j]
            stock2_data = dataset.get_ticker_data(stock2)
            
            #Compute correlation for the last available closing prices only
            correlation = compute_daily_correlation(stock1_data, stock2_data, date)
            print(correlation)
            
            if correlation is not None:
                G.add_edge(stock1, stock2, weight=correlation)
    
    return G