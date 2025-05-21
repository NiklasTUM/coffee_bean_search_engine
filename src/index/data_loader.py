import os
from logging import Logger

import pandas as pd
from langchain_core.documents import Document
from src.constants import constants
from src.logger.CustomLogger import CustomLogger

import nltk


class DataLoader:
    def __init__(self, logger: Logger = None):
        """
        Initializes the DataLoader with the specified directory and a logger.

        Args:
            logger (logger, optional): logger instance for logging. If not provided, a default logger is set up.
        """
        self.dataset_path = os.path.join(constants.root_dir, "data", "simplified_coffee.csv")
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = logger or CustomLogger(self.log_dir, "logs.log").logger

    def load_coffee_data(self) -> pd.DataFrame:
        """
        Loads a CSV file with columns:
        name, roaster, roast, loc_country, origin, 100g_USD, rating, review_date, review

        Handles quoted fields (especially the 'review' column, which may contain commas).
        """
        df = pd.read_csv(self.dataset_path, quotechar='"', encoding='utf-8')
        return df


if __name__ == "__main__":
    logger_instance = CustomLogger().logger
    data_loader = DataLoader(logger=logger_instance)
    df = data_loader.load_coffee_data()

    # Print the first row
    if not df.empty:
        print("First row of the coffee dataset:\n")
        print(df.iloc[0])
    else:
        print("The dataset is empty.")
