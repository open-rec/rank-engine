from algorithm.rank.lr import LRModel
from algorithm.rank.fm import FMModel
from algorithm.rank.lightgbm import LightGBMBinaryModel

model_func_map = {
    "lr": LRModel,
    "fm": FMModel,
    "lightgbm": LightGBMBinaryModel,
}
