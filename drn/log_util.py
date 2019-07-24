import logging

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"

logger = None


def setup_logger(name, level=logging.DEBUG):
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def get_logger():
    if logger is None:
        print("Setup logger first.")
        return

    return logger
