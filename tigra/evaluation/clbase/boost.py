from tigra.evaluation.clbase.skl import SKLCModel

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


class XGB(SKLCModel):
    def __init__(self, n_estimators=4):
        super().__init__("XGB", XGBClassifier(n_estimators=n_estimators, objective='binary:logistic'))
  

class CAT(SKLCModel):
    def __init__(self):
        super().__init__("CAT", CatBoostClassifier(logging_level='Silent'))
    

class LGBM(SKLCModel):
    def __init__(self):
        super().__init__("LGBM", LGBMClassifier(verbose=-1))
