import logging
import os


class RAGLogger:
    def __init__(self, log_dir: str = 'logs', log_file: str = 'RAG.log'):
        """
        Initializes the RAGLogger instance and sets up the logger.

        Args:
            log_dir (str): Directory where the log file will be stored.
            log_file (str): Name of the log file.
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Sets up a logger with console and file handlers.
        Automatically creates the log directory if it does not exist.

        Returns:
            Logger: The logger with console and file handlers.
        """
        logger = logging.getLogger(__name__)

        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.DEBUG)

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        console_handler = logging.StreamHandler()
        log_file_path = os.path.join(self.log_dir, self.log_file)
        file_handler = logging.FileHandler(log_file_path)

        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger
