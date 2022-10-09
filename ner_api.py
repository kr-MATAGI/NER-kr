import json

import torch
from eunjeon import Mecab
from model.electra_lstm_crf import ELECTRA_POS_LSTM
from transformers import ElectraTokenizer, ElectraModel

#===========================================================================
def conv_mecab_tag_to_nikl(mecab_pos_list):
#===========================================================================
    ret_mecab_pos = []
    for mecab_pos in mecab_pos_list:
        if "NNBC" in mecab_pos[1]:
            new_mecab_pos = (mecab_pos[0], mecab_pos[1].replace("NNBC", "NNB"))
        elif


#===========================================================================
def make_pos_tag_ids(src_sent, mecab, tokenizer, input_ids):
#===========================================================================
    ret_pos_tag_ids = None

    mecab_res = mecab.pos(src_sent)
    mecab_res = conv_mecab_tag_to_nikl(mecab_res)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(mecab_res)
    print(tokens)

    return ret_pos_tag_ids

#### MAIN ####
if "__main__" == __name__:
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