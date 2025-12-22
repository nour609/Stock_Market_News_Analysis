import pandas as pd
import requests
from pathlib import Path

class StockNewsDataLoader:
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        
    def download_dataset(self, url):
        """
        Download the dataset from Kaggle URL
        Note: You'll need to download the dataset manually from:
        https://www.kaggle.com/datasets/aaron7sun/stocknews/data
        and place it in the data directory
        """
        print("Please download the dataset manually from:")
        print("https://www.kaggle.com/datasets/aaron7sun/stocknews/data")
        print(f"And place it in the {self.data_path} directory")
        
    def load_data(self, filename):
        """
        Load the dataset from local storage
        """
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        return pd.read_csv(file_path)