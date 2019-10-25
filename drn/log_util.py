import os
import sys
import logging
from datetime import datetime


class DRNLogger(object):
    """docstring for DRNLogger"""

    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"

    def __init__(self, name, filepath=".", level=logging.DEBUG):
        log_name = "{name}_{date}.log".format(name=name, date=datetime.now().strftime("%Y-%m-%d_%H%M%S"))

        if filepath is not None:
            logging.basicConfig(filename=os.path.join(filepath, log_name),
                                filemode='a',
                                format=self.FORMAT)
        else:
            logging.basicConfig(format=self.FORMAT)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

    def get_logger(self):
        return self.logger
