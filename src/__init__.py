import logging


class MyFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record)


def configure_logging():
    console_handler = logging.StreamHandler()

    formatter = MyFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(console_handler)


configure_logging()
