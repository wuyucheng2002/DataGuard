from .data_prepare import Data as Data_TIPRDC
from .component import Feature_Extractor, Classifier, MutlInfo
from .pretrain import train_FE_CF
from .main import get_FE, test_downstream_task