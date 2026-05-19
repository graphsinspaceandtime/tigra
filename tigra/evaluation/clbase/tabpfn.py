from tigra.evaluation.clbase.skl import SKLCModel

from tabpfn import TabPFNClassifier


class TPFN(SKLCModel):
    def __init__(self):
        super().__init__("TabPFN", TabPFNClassifier())
