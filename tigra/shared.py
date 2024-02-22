from pathlib import Path
import os


def get_root() -> Path:
    """
    Get relative path to the root of the library. Root is looked up using the
    "tests" directory as a stop gap.
    """
    path = Path(__file__)
    while os.path.basename(path) != "tests":
        path = path.parent
    path = path.parent
    return path


def get_data_prefix() -> Path:
    """
    Get relative path prefix for the "data" directory.
    """
    return get_root() / "data"
