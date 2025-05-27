import os
import re
from logging import Logger

import pandas as pd
from src.constants import constants
from src.logger.custom_logger import CustomLogger


class DataLoader:
    def __init__(self):
        """
        Initializes the DataLoader with the specified directory and a logger.

        """
        self.dataset_path = os.path.join(constants.root_dir, "data", "coffee_analysis.csv")
        self.log_dir = os.path.join(constants.root_dir, "logs")
        self.logger = CustomLogger(self.log_dir, "logs.log").logger

    def load_coffee_data(self) -> pd.DataFrame:
        """
        Loads a CSV file with columns:
        name, roaster, roast, loc_country, origin, 100g_USD, rating, review_date, review

        Handles quoted fields (especially the 'review' column, which may contain commas).
        """
        df = pd.read_csv(self.dataset_path, quotechar='"', encoding='utf-8')
        return df

    def calculate_hyperlink_percentage(self, df: pd.DataFrame) -> float:
        """
        Analyzes the DataFrame and calculates the percentage of rows that contain
        at least one hyperlink in 'desc_1', 'desc_2', or 'desc_3'.

        Args:
            df (pd.DataFrame): The coffee dataset.

        Returns:
            float: Percentage of rows with at least one hyperlink.
        """
        hyperlink_pattern = re.compile(r'www\.\S+', re.IGNORECASE)
        count_with_links = 0
        total_rows = len(df)

        for _, row in df.iterrows():
            if any(
                    isinstance(row.get(col), str) and hyperlink_pattern.search(row[col])
                    for col in ['desc_1', 'desc_2', 'desc_3']
            ):
                count_with_links += 1

        percentage = (count_with_links / total_rows * 100) if total_rows > 0 else 0.0
        print(f"Rows with hyperlinks: {count_with_links} / {total_rows} ({percentage:.2f}%)")
        return percentage


if __name__ == "__main__":
    logger_instance = CustomLogger().logger
    data_loader = DataLoader(logger=logger_instance)
    df = data_loader.load_coffee_data()
    data_loader.calculate_hyperlink_percentage(df)

    # Print the first row
    if not df.empty:
        print("First row of the coffee dataset:\n")
        print(df.iloc[0])
    else:
        print("The dataset is empty.")
