import typing
from tigra.core.dgraph import DynamicSparseMatrixGraph


class DynamicAutoEncoderEmbeddingModel:
    def __init__(self) -> None:
        pass

    def get_method_name(self) -> str:
        raise Exception("Method not implemented.")

    def learn_embedding(self, data: DynamicSparseMatrixGraph):
        raise Exception("Method not implemented.")

    def get_embedding(self) -> typing.List[any]:
        raise Exception("Method not implemented.")
