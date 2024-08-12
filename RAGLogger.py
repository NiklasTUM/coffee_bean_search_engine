import logging
import os


class RAGLogger:
    """
    A class to set up and manage logging for the RAG system, with both console and file handlers.

    Attributes:
        log_dir (str): Directory where the log file will be stored.
        log_file (str): Name of the log file.
        logger (logging.Logger): Configured logger instance for logging information and errors.
    """

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
        self.logger.info("RAGLogger initialized successfully.")

    def _setup_logger(self) -> logging.Logger:
        """
        Sets up a logger with console and file handlers.
        Automatically creates the log directory if it does not exist.

        Returns:
            logging.Logger: The configured logger with console and file handlers.
        """
        logger = logging.getLogger(__name__)

        # Check if the logger already has handlers (to prevent adding them multiple times)
        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.DEBUG)
        self.logger.info(f"Setting up logger with log file at {self.log_dir}/{self.log_file}.")

        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger.info(f"Log directory {self.log_dir} created/verified.")

        # Set up console handler (INFO level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Set up file handler (DEBUG level)
        log_file_path = os.path.join(self.log_dir, self.log_file)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Define a common formatter for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        self.logger.info("Logger setup completed with console and file handlers.")

        return logger
