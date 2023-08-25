import logging, sys

def get(name, level=logging.INFO):
    logging.root.setLevel(logging.NOTSET)
    # Create handlers
    c_handler = logging.StreamHandler()
    # f_handler = logging.FileHandler('file.log')
    c_handler.setLevel(level)
    # f_handler.setLevel(logging.ERROR)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('[%(name)s - %(levelname)s] - [%(processName)s / %(threadName)s] - %(message)s')
    # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    # f_handler.setFormatter(f_format)
    l = logging.getLogger(name)
    l.addHandler(c_handler)
    return l
