import duckdb


def connect(path: str):
    return duckdb.connect(path)
