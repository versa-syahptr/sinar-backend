import logging, sys

# Create a custom logger
# logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
# f_handler = logging.FileHandler('file.log')
c_handler.setLevel(logging.DEBUG)
# f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter('[%(name)s] [%(levelname)s] [thread: %(thread)d]- %(message)s')
# f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
# f_handler.setFormatter(f_format)

# Add handlers to the logger
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)
def get(name):
    l = logging.getLogger(name)
    l.addHandler(c_handler)
    return l
