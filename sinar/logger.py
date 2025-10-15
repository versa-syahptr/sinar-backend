import logging
import colorlog
import types

format = '%(log_color)s[%(levelname)s @ %(name)s]%(reset)s - [%(processName)s / %(threadName)s] - %(message)s'

formatter = colorlog.ColoredFormatter(
    format,
    log_colors={
        'DEBUG': 'yellow',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    reset=True,
    style='%'
)

# === ADD TRACE LEVEL ===
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

def attach_trace(logger_obj):
    logger_obj.trace = types.MethodType(trace, logger_obj)
    
# global handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# global logger
logger = logging.getLogger("sinar")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
logger.trace = trace


def set_level(level_name):
    """Change the global SINAR log level at runtime."""
    level_name = level_name.lower()
    if level_name == "trace":
        level = TRACE_LEVEL_NUM
    else:
        level = getattr(logging, level_name.upper(), None)

    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    logger.setLevel(level)
    handler.setLevel(level)
    logger.info(f"Log level set to {level_name.upper()}")

def get(name):
    # Remove top-level package name if it matches
    if name.startswith("sinar."):
        name = name[len("sinar."):]
    child = logger.getChild(name)
    attach_trace(child)
    return child


