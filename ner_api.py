import json
import datetime as dt
import copy

import torch
from eunjeon import Mecab
from model.electra_lstm_crf import ELECTRA_POS_LSTM
from transformers import ElectraTokenizer, ElectraModel
from utils.tag_def import ETRI_TAG

from dataclasses import dataclass, field
from typing import List

#### Dataclass
@dataclass
class Res_ne:
    id: str = ""
    word: str = ""
    label: str = ""
    begin: int = ""
    end: int = ""

@dataclass
class Res_results:
    id: str = ""
    text: str = ""
    ne_list: List[Res_ne] = field(default_factory=list)


#===========================================================================
def conv_mecab_tag_to_nikl(mecab_pos_list):
#===========================================================================
    pass

#===========================================================================
def make_response_json(res_data: Res_results):
#===========================================================================
    json_dict = {
        "date": "",
        "text": "",
        "ne": []
    }

    json_dict["date"] = dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f ")
    json_dict["text"] = res_data.text
    for ne_idx, ne_item in enumerate(res_data.ne_list):
        res_ne = {
            "id": ne_item.id, "word": ne_item.word, "label": ne_item.label,
            "begin": ne_item.begin, "end": ne_item.end
        }
        json_dict["ne"].append(copy.deepcopy(res_ne))
    json_string = json.dumps(json_dict, ensure_ascii=False)
    return json_string

#===========================================================================
def make_pos_tag_ids(src_sent, mecab, tokenizer, input_ids):
#===========================================================================
    # [ (word, [(word, [pos,...]), ...] ]
    # -> [ (word, [(tokens, [pos,...]), ...] ]

    tokenized_mecab_list = []
    for m_idx, mecab_pair in enumerate(mecab_pair_list):
        new_pos_pair_list = []
        for m_pos_idx, m_pos_pair in enumerate(mecab_pair[1]):
            tokenized_word = tokenizer.tokenize(m_pos_pair[0])
            new_pos_pair_list.append(Tok_Pos(tokens=tokenized_word,
                                             pos=m_pos_pair[1]))
        tokenized_mecab_list.append(Mecab_Item(word=mecab_pair[0], tok_pos_list=new_pos_pair_list))

    return ret_pos_tag_ids

#### MAIN ####
if "__main__" == __name__:
    '''
        id: str = ""
        word: str = ""
        label: str = ""
        begin: int = ""
        end: int = ""
    '''
    # test_res_ne_1 = Res_ne(id="1", word="대통령", label="CV", begin=1, end=3)
    # test_res_ne_2 = Res_ne(id="2", word="최재훈", label="PS", begin=5, end=8)
    # test_merge_ne = [test_res_ne_1, test_res_ne_2]
    # test_results = Res_results(id="1", text=" 대통령 최재훈", ne_list=test_merge_ne)
    # make_response_json(test_results)
    # exit()

    model_path = "./test_model"

    # Init
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = ELECTRA_POS_LSTM.from_pretrained(model_path)
    mecab = Mecab()

    while True:
        print("입력 문장: ")
        input_sent = str(input())

        token_res = tokenizer(input_sent)
        pos_tag_ids = make_pos_tag_ids(src_sent=input_sent, mecab=mecab,
                                       tokenizer=tokenizer, input_ids=token_res["input_ids"])

        break
        inputs = {
            "input_ids": torch.LongTensor([token_res["input_ids"]]),
            "attention_mask": torch.LongTensor([token_res["attention_mask"]]),
            "token_type_ids": torch.LongTensor([token_res["token_type_ids"]]),
            "pos_tag_ids": torch.LongTensor([pos_tag_ids])
        }