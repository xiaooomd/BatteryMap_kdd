"""Centralized model construction for prediction entry points."""

from models import Autoformer
from models import BiGRU
from models import BiLSTM
from models import CNN
from models import CPBiGRU
from models import CPBiLSTM
from models import CPGRU
from models import CPLSTM
from models import CPMLP
from models import CPTransformer
from models import DLinear
from models import GRU
from models import LSTM
from models import MICN
from models import MLP
from models import PatchTST
from models import Transformer
from models import iTransformer
from models import DACNet
from models import DegradeBEATS
from models import TrendSpec


MODEL_REGISTRY = {
    'Transformer': Transformer.Model,
    'CPBiLSTM': CPBiLSTM.Model,
    'CPBiGRU': CPBiGRU.Model,
    'CPGRU': CPGRU.Model,
    'CPLSTM': CPLSTM.Model,
    'BiLSTM': BiLSTM.Model,
    'BiGRU': BiGRU.Model,
    'LSTM': LSTM.Model,
    'GRU': GRU.Model,
    'PatchTST': PatchTST.Model,
    'iTransformer': iTransformer.Model,
    'DLinear': DLinear.Model,
    'CPMLP': CPMLP.Model,
    'Autoformer': Autoformer.Model,
    'MLP': MLP.Model,
    'MICN': MICN.Model,
    'CNN': CNN.Model,
    'CPTransformer': CPTransformer.Model,
    'DACNet': DACNet.Model,
    'DegradeBEATS': DegradeBEATS.Model,
    'TrendSpec': TrendSpec.Model,
}


def build_model(model_name, args):
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(sorted(MODEL_REGISTRY))
        raise Exception(f"The {model_name} is not an implemented baseline! Available: {available}")
    return MODEL_REGISTRY[model_name](args).float()
