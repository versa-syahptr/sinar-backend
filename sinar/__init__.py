import os
from contextlib import contextmanager


@contextmanager
def _stderr_silenced():
    """Point fd 2 at /dev/null: TF >= 2.20 prints oneDNN/CUDA notes from C++ before absl logging
    is initialised, so TF_CPP_MIN_LOG_LEVEL can't filter them."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


# import TensorFlow once, quietly, before any sinar module does; TF_CPP_MIN_LOG_LEVEL=0 shows the notes
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.environ["TF_CPP_MIN_LOG_LEVEL"] != "0":
    with _stderr_silenced():
        import tensorflow  # noqa: F401
del _stderr_silenced

from sinar.core import SINAR  # noqa: E402
