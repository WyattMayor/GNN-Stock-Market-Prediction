import pandas as pd
import os
from typing import Optional

class NASDAQDataset:
    def __init__(self, base_dir: str, metadata_file: Optional[str] = None):
        """
        Initializes the NASDAQDataset class, loading all data for stocks and ETFs at once.

        Args:
            base_dir (str): Base directory path where the dataset folder (containing etf, stocks) is located.
            metadata_file (str, optional): Path to the metadata CSV file (symbols_valid_meta.csv). Defaults to None.
        """
        self.data_dir = os.path.join(base_dir, "dataset")
        self.metadata_file = metadata_file
        self.metadata = self.load_metadata() if metadata_file else None
        self.data = self.load_all_data()

    def load_metadata(self) -> pd.DataFrame:
        """Loads metadata for each ticker symbol from the metadata CSV file."""
        return pd.read_csv(self.metadata_file)

    def load_all_data(self) -> pd.DataFrame:
        """
        Loads all data from both stocks and ETFs folders and combines them into a single DataFrame.

        Returns:
            pd.DataFrame: Combined DataFrame with all ticker data from stocks and ETFs.
        """
        all_data = []
        for folder in ["stocks", "etf"]:
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".csv"):
                        ticker = filename.split(".")[0]
                        file_path = os.path.join(folder_path, filename)
                        data = pd.read_csv(file_path, parse_dates=["Date"])
                        data["Ticker"] = ticker
                        data["Type"] = folder
                        all_data.append(data)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves data for a specific ticker from the loaded dataset.

        Args:
            ticker (str): Ticker symbol to retrieve.

        Returns:
            pd.DataFrame: DataFrame with data for the specified ticker.
        """
        return self.data[self.data["Ticker"] == ticker].copy() if not self.data.empty else pd.DataFrame()

    def get_metadata_info(self, ticker: str) -> Optional[pd.Series]:
        """
        Retrieves metadata information for a specific ticker.

        Args:
            ticker (str): Ticker symbol.

        Returns:
            pd.Series or None: Metadata information for the ticker if available.
        """
        if self.metadata is not None:
            return self.metadata[self.metadata['Symbol'] == ticker].squeeze()
        return None
    