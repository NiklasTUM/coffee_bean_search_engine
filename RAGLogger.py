import logging
from logging import Logger


class RAGLogger(Logger):

    @staticmethod
    def setup_logger() -> Logger:
        """
        Sets up a logger with console and file handlers.

        Returns:
            Logger: The logger with console and file handlers.
        """
        logger = logging.getLogger(__name__)
        if logger.hasHandlers():
            return logger

        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('logs/apicaller.log')

        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger
