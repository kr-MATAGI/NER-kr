import numpy as np
import torch
import pickle
import pandas as pd

from dataclasses import dataclass
from collections import deque, defaultdict
from typing import List

import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/NER_Private")
from transformers import ElectraTokenizer
from model.electra_eojeol_model import Electra_Eojeol_Model
from model.electra_lstm_crf import ELECTRA_POS_LSTM
from utils.tag_def import ETRI_TAG, NIKL_POS_TAG

from seqeval.metrics.sequence_labeling import get_entities


#### Dataclass
@dataclass
class ERR_OUTPUT:
    data_idx: int = -1
    sent_text: str = ""
    ne_label: str = ""
    ne_pred: str = ""
    error_cate


#===========================================================================
def
#===========================================================================

#### MAIN ###
if "__main__" == __name__:
    print("[analyze_outputs] __MAIN__ !")