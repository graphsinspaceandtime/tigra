import typing


class DynamicAutoEncoderEmbeddingModel:
    def __init__(self, d, **kwargs) -> None:
        self._d = d
        hyper_params = {
            "actfn": "relu",
        }
        hyper_params.update(kwargs)
        for key, value in hyper_params.items():
            self.__setattr__(f"_{key}", value)

    def get_method_name(self) -> str:
        return self._method_name

    def learn_embedding(self, graphs: list):
        raise Exception("Method not implemented.")

    def get_embedding(self) -> typing.List[any]:
        raise Exception("Method not implemented.")
