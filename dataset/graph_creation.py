import networkx as nx
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict

def create_time_series_daily_nodes_network(json_file: str) -> nx.Graph:
    """
    Creates a NetworkX graph from a JSON file where each date for a stock is a separate node,
    and edges represent correlations between stocks on the same date.

    Args:
        json_file (str): Path to the JSON file with correlation data in the format:
                         [{"Date1": str, "Stock1": str, "Date2": str, "Stock2": str, "Correlation": float}, ...]

    Returns:
        nx.Graph: A NetworkX graph where each node is a specific date for a stock,
                  and edges represent daily correlations between stocks.
    """
    G = nx.Graph()

    with open(json_file, "r") as f:
        correlation_data = json.load(f)
    
    for entry in correlation_data:
        date1 = entry["Date1"]
        stock1 = entry["Stock1"]
        date2 = entry["Date2"]
        stock2 = entry["Stock2"]
        correlation = entry["Correlation"]

        node1 = f"{stock1}_{date1}"
        node2 = f"{stock2}_{date2}"

        if not G.has_node(node1):
            G.add_node(node1, stock=stock1, date=date1)
        if not G.has_node(node2):
            G.add_node(node2, stock=stock2, date=date2)

        G.add_edge(node1, node2, weight=correlation)

    return G

def timeline_layout(G: nx.Graph):
    """
    Creates a timeline layout for the NetworkX graph, arranging nodes along the x-axis by date
    and the y-axis by stock.

    Args:
        G (nx.Graph): The NetworkX graph with date and stock attributes.

    Returns:
        dict: A dictionary of positions keyed by node.
    """
    pos = {}
    stocks = list({data["stock"] for _, data in G.nodes(data=True)})

    stock_y_positions = {stock: i for i, stock in enumerate(stocks)}

    for node, data in G.nodes(data=True):
        date = datetime.strptime(data["date"], "%Y-%m-%d")
        x = date.timestamp()
        y = stock_y_positions[data["stock"]]

        pos[node] = (x, y)
    
    return pos

def visualize_timeline_network(G: nx.Graph):
    """
    Visualizes the NetworkX graph using a timeline layout, with each node as a specific date for a stock,
    and edges representing correlations between stocks on the same date.

    Args:
        G (nx.Graph): The NetworkX graph with nodes representing specific dates for stocks and edges as correlations.
    """
 
    pos = timeline_layout(G)

    plt.figure(figsize=(14, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, font_weight="bold")

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Daily Stock Correlation Network (Timeline Layout)")
    plt.xlabel("Time")
    plt.ylabel("Stocks")
    plt.show()