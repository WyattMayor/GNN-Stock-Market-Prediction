import pandas as pd
import numpy as np
import json
from typing import List, Optional

def compute_adjusted_cross_correlation(stock1_data: pd.DataFrame, stock2_data: pd.DataFrame, 
                                       start_date: Optional[str] = None, end_date: Optional[str] = None, 
                                       decay_factor: float = 0.95) -> List[dict]:
    correlations = []

    if start_date:
        stock1_data = stock1_data[stock1_data['Date'] >= pd.to_datetime(start_date)]
        stock2_data = stock2_data[stock2_data['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        stock1_data = stock1_data[stock1_data['Date'] <= pd.to_datetime(end_date)]
        stock2_data = stock2_data[stock2_data['Date'] <= pd.to_datetime(end_date)]

    dates1 = stock1_data['Date'].values
    dates2 = stock2_data['Date'].values
    prices1 = stock1_data[['Open', 'High', 'Low', 'Close']].values
    prices2 = stock2_data[['Open', 'High', 'Low', 'Close']].values
    
    for idx1, date1 in enumerate(dates1):
        day_data1 = prices1[idx1]

        for idx2, date2 in enumerate(dates2):
            if date2 >= date1:
                day_data2 = prices2[idx2]

                correlation = np.corrcoef(day_data1, day_data2)[0, 1]
                
                days_apart = abs((date2 - date1).astype('timedelta64[D]').astype(int))
                adjusted_correlation = correlation * (decay_factor ** days_apart)

                correlations.append({
                    "Date1": pd.to_datetime(date1).strftime('%Y-%m-%d'),
                    "Stock1": stock1_data['Ticker'].iloc[0],
                    "Date2": pd.to_datetime(date2).strftime('%Y-%m-%d'),
                    "Stock2": stock2_data['Ticker'].iloc[0],
                    "Correlation": adjusted_correlation
                })

    return correlations

def create_cross_date_correlation_graph(dataset, stock_list: List[str], 
                                        start_date: Optional[str] = None, end_date: Optional[str] = None, 
                                        output_file: str = "correlation_data.json"):
    """
    Generates a cross-date correlation graph of daily adjusted correlations between multiple stocks and stores it in a JSON file.

    Args:
        dataset (NASDAQDataset): NASDAQDataset instance containing loaded stock data.
        stock_list (List[str]): List of stock tickers to include in the graph.
        start_date (str, optional): Start date for filtering data in 'YYYY-MM-DD' format.
        end_date (str, optional): End date for filtering data in 'YYYY-MM-DD' format.
        output_file (str): Name of the JSON file to store the results.
    """
    all_correlations = []

    for i, stock1 in enumerate(stock_list):
        stock1_data = dataset.get_ticker_data(stock1)
        
        for j in range(i + 1, len(stock_list)):
            stock2 = stock_list[j]
            stock2_data = dataset.get_ticker_data(stock2)
            
            correlations = compute_adjusted_cross_correlation(stock1_data, stock2_data, start_date, end_date)
            all_correlations.extend(correlations)
    
    with open(output_file, "w") as f:
        json.dump(all_correlations, f, indent=4)
