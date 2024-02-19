from pathlib import Path


def get_root() -> Path:
    return Path(__file__).parent


def get_data_prefix() -> Path:
    return get_root() / "data"
