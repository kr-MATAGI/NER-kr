import copy
import random
import numpy as np
import pickle

from dataclasses import dataclass, field
from collections import deque

import sacrebleu
from eunjeon import Mecab

from tag_def import ETRI_TAG, NIKL_POS_TAG, MECAB_POS_TAG
from data_def import Sentence, NE, Morp, Word

from gold_corpus_npy_maker import (
    conv_TTA_ne_category
)

from typing import List, Dict, Tuple
from transformers import ElectraTokenizer

# SPAN NER Package
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from torch.utils.data import Dataset

#==========================================================================================
def load_ne_entity_list(src_path: str = ""):
#==========================================================================================
    all_sent_list = []

    with open(src_path, mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[mecab_npy_maker][load_ne_entity_list] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)

    return all_sent_list

#=======================================================================================
def convert2tokenIdx(tokens, all_span_idxs, span_idxLab, max_len):
#=======================================================================================
    # convert the all the span_idxs from word-level to token-level
    sidxs = [x1 for (x1, x2) in all_span_idxs]
    eidxs = [x2 for (x1, x2) in all_span_idxs]

    span_idxs_new_label = {}
    for ns, ne, ose in zip(sidxs, eidxs, all_span_idxs):
        os, oe = ose
        oes_str = "{};{}".format(os, oe)
        nes_str = "{};{}".format(ns, ne)
        if oes_str in span_idxLab:
            label = span_idxLab[oes_str]
            span_idxs_new_label[nes_str] = label
        else:
            span_idxs_new_label[nes_str] = 'O'

    return span_idxs_new_label


#=======================================================================================
def make_span_idx_label_pair(ne_list, text_tokens):
#=======================================================================================
    ret_dict = {}

    # print(text_tokens)
    b_check_use = [False for _ in range(len(text_tokens))]
    for ne_idx, ne_item in enumerate(ne_list):
        ne_char_list = list(ne_item.text.replace(" ", ""))

        concat_item_list = []
        for tok_idx in range(len(text_tokens)):
            if b_check_use[tok_idx]:
                continue
            for sub_idx in range(tok_idx + 1, len(text_tokens)):
                concat_word = ["".join(x).replace("##", "") for x in text_tokens[tok_idx:sub_idx]]
                concat_item_list.append(("".join(concat_word), (tok_idx, sub_idx - 1))) # Modify -1
        concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
        concat_item_list.sort(key=lambda x: len(x[0]))
        if 0 >= len(concat_item_list):
            continue
        target_idx_pair = concat_item_list[0][1]
        key = str(target_idx_pair[0]) + ";" + str(target_idx_pair[1])
        ret_dict[key] = ne_item.type

    # print(ret_dict)
    return ret_dict

#=======================================================================================
def make_span_npy(tokenizer_name: str, src_list: List[Sentence],
                  seq_max_len: int = 128, debug_mode: bool = False,
                  span_max_len: int = 10, save_npy_path: str = None):
#=======================================================================================
    span_minus = int((span_max_len + 1) * span_max_len / 2)
    max_num_span = int(seq_max_len * span_max_len - span_minus)

    npy_dict = {
        "input_ids": [],
        "label_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "all_span_len_list": [],
        "real_span_mask_token": [],
        "span_only_label_token": [],
        "all_span_idx_list": []
    }

    # shuffle
    random.shuffle(src_list)

    # Init
    # etri_tags = sorted(list(set([k.replace("B-", "").replace("I-", "") for k, v in ETRI_TAG.items()])),
    #                    key=lambda x: len(x))
    etri_tags = {'O': 0, 'FD': 1, 'EV': 2, 'DT': 3, 'TI': 4, 'MT': 5,
                 'AM': 6, 'LC': 7, 'CV': 8, 'PS': 9, 'TR': 10,
                 'TM': 11, 'AF': 12, 'PT': 13, 'OG': 14, 'QT': 15}
    ne_ids2tag = {v: i for i, v in enumerate(etri_tags)}
    ne_detail_ids2_tok = {v: k for k, v in ETRI_TAG.items()}
    print(ne_ids2tag)

    mecab = Mecab()
    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    for proc_idx, src_item in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")

        # Mecab
        mecab_res = mecab.pos(src_item.text)
        text_tokens = []
        for mecab_item in mecab_res:
            tokens = tokenizer.tokenize(mecab_item[0])
            text_tokens.extend(tokens)

        # Text Tokens
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        if seq_max_len <= len(text_tokens):
            text_tokens = text_tokens[:seq_max_len - 1]
            text_tokens.append("[SEP]")

            valid_token_len = seq_max_len
        else:
            text_tokens.append("[SEP]")
            valid_token_len = len(text_tokens)
        
        # NE - Token 단위
        label_ids = [ETRI_TAG["O"]] * len(text_tokens)
        b_check_use = [False for _ in range(len(text_tokens))]
        for ne_idx, ne_item in enumerate(src_item.ne_list):
            ne_char_list = list(ne_item.text.replace(" ", ""))
            concat_item_list = []
            for tok_idx in range(len(text_tokens)):
                if b_check_use[tok_idx]:
                    continue
                for sub_idx in range(tok_idx + 1, len(text_tokens)):
                    concat_word = ["".join(x).replace("##", "") for x in text_tokens[tok_idx:sub_idx]]
                    concat_item_list.append(("".join(concat_word), (tok_idx, sub_idx)))
            concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
            concat_item_list.sort(key=lambda x: len(x[0]))
            if 0 >= len(concat_item_list):
                continue
            target_idx_pair = concat_item_list[0][1]

            for bio_idx in range(target_idx_pair[0], target_idx_pair[1]):
                b_check_use[bio_idx] = True
                if bio_idx == target_idx_pair[0]:
                    label_ids[bio_idx] = ETRI_TAG["B-" + ne_item.type]
                else:
                    label_ids[bio_idx] = ETRI_TAG["I-" + ne_item.type]

        if seq_max_len <= len(label_ids):
            label_ids = label_ids[:seq_max_len - 1]
            label_ids.append(ETRI_TAG["O"])
        else:
            label_ids_size = len(label_ids)
            for _ in range(seq_max_len - label_ids_size):
                label_ids.append(ETRI_TAG["O"])

        # Span NER
        context_split = [x[0] for x in mecab_res]
        all_span_idx_list = enumerate_spans(context_split, offset=0, max_span_width=span_max_len)
        all_span_len_list = []
        for idx_pair in all_span_idx_list:
            s_idx, e_idx = idx_pair
            span_len = e_idx - s_idx + 1
            all_span_len_list.append(span_len)

        span_idx_label_dict = make_span_idx_label_pair(src_item.ne_list, text_tokens)
        span_idx_new_label_dict = convert2tokenIdx(tokens=text_tokens, span_idxLab=span_idx_label_dict,
                                                   all_span_idxs=all_span_idx_list, max_len=seq_max_len) # or Max Len
        span_only_label_token = []
        for idx_str, label in span_idx_new_label_dict.items():
            span_only_label_token.append(ne_ids2tag[label])

        text_tokens += ["[PAD]"] * (seq_max_len - valid_token_len)
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = ([1] * valid_token_len) + ([0] * (seq_max_len - valid_token_len))
        token_type_ids = [0] * seq_max_len

        # Span Len
        all_span_idx_list = all_span_idx_list[:max_num_span]
        span_only_label_token = span_only_label_token[:max_num_span]
        all_span_len_list = all_span_len_list[:max_num_span]
        real_span_mask_token = np.ones_like(span_only_label_token).tolist()

        '''
            all_span_idx_list와 span_only_label_token을 통해서
            정답 span을 알 수 있다.
            실제적으로 Span을 예측하고 Label을 Return하는 형식으로 output이 나올 것이라
            Span을 text token으로 변환할때 BIO 태깅이 들어가야 할듯.
        '''

        # print(span_only_label_token)
        # print(all_span_len_list)
        # print(real_span_mask_token)
        # print(all_span_idx_list)

        if max_num_span > len(span_only_label_token):
            diff_len = max_num_span - len(span_only_label_token)
            span_only_label_token += [0] * diff_len
        if max_num_span > len(all_span_len_list):
            diff_len = max_num_span - len(all_span_len_list)
            all_span_len_list += [0] * diff_len
        if max_num_span > len(real_span_mask_token):
            diff_len = max_num_span - len(real_span_mask_token)
            real_span_mask_token += [0] * diff_len
        if max_num_span > len(all_span_idx_list):
            diff_len = max_num_span - len(all_span_idx_list)
            all_span_idx_list += [(0, 0)] * diff_len

        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["label_ids"].append(label_ids)

        npy_dict["span_only_label_token"].append(span_only_label_token)
        npy_dict["all_span_len_list"].append(all_span_len_list)
        npy_dict["real_span_mask_token"].append(real_span_mask_token)
        npy_dict["all_span_idx_list"].append(all_span_idx_list)

        if debug_mode:
            for t, l in zip(text_tokens, label_ids):
                print(t, ne_detail_ids2_tok[l])
            input()
        # if 0 == (proc_idx % 100):
        #     # For save Test
        #     break

    save_span_npy(npy_dict, len(src_list), save_npy_path)

#=======================================================================================
def save_span_npy(npy_dict, src_list_len, save_path):
#=======================================================================================
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["label_ids"] = np.array(npy_dict["label_ids"])

    npy_dict["span_only_label_token"] = np.array(npy_dict["span_only_label_token"])
    npy_dict["all_span_len_list"] = np.array(npy_dict["all_span_len_list"])
    npy_dict["real_span_mask_token"] = np.array(npy_dict["real_span_mask_token"])
    npy_dict["all_span_idx_list"] = np.array(npy_dict["all_span_idx_list"])

    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"label_ids.shape: {npy_dict['label_ids'].shape}")

    print(f"span_only_label_token.shape: {npy_dict['span_only_label_token'].shape}")
    print(f"all_span_len_list.len: {npy_dict['all_span_len_list'].shape}")
    print(f"real_span_mask_token.len: {npy_dict['real_span_mask_token'].shape}")
    print(f"all_span_idx_list.len: {npy_dict['all_span_idx_list'].shape}")

    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    dev_size = train_size + split_size

    # Train
    train_np = [npy_dict["input_ids"][:train_size],
                npy_dict["attention_mask"][:train_size],
                npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_label_ids_np = npy_dict["label_ids"][:train_size]

    train_span_only_label_token_np = npy_dict["span_only_label_token"][:train_size]
    train_all_span_len_list_np = npy_dict["all_span_len_list"][:train_size]
    train_real_span_mask_token_np = npy_dict["real_span_mask_token"][:train_size]
    train_all_span_idx_list_np = npy_dict["all_span_idx_list"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_label_ids.shape: {train_label_ids_np.shape}")
    print(f"train_span_only_label_token_np.shape: {train_span_only_label_token_np.shape}")
    print(f"train_all_span_len_list_np.len: {train_all_span_len_list_np.shape}")
    print(f"train_real_span_mask_token_np.len: {train_real_span_mask_token_np.shape}")
    print(f"train_all_span_idx_list_np.len: {train_all_span_idx_list_np.shape}")

    # Dev
    dev_np = [npy_dict["input_ids"][train_size:dev_size],
              npy_dict["attention_mask"][train_size:dev_size],
              npy_dict["token_type_ids"][train_size:dev_size]
              ]
    dev_np = np.stack(dev_np, axis=-1)
    dev_label_ids_np = npy_dict["label_ids"][train_size:dev_size]

    dev_span_only_label_token_np = npy_dict["span_only_label_token"][train_size:dev_size]
    dev_all_span_len_list_np = npy_dict["all_span_len_list"][train_size:dev_size]
    dev_real_span_mask_token_np = npy_dict["real_span_mask_token"][train_size:dev_size]
    dev_all_span_idx_list_np = npy_dict["all_span_idx_list"][train_size:dev_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_label_ids.shape: {dev_label_ids_np.shape}")

    print(f"dev_span_only_label_token_np.shape: {dev_span_only_label_token_np.shape}")
    print(f"dev_all_span_len_list_np.len: {dev_all_span_len_list_np.shape}")
    print(f"dev_real_span_mask_token_np.len: {dev_real_span_mask_token_np.shape}")
    print(f"dev_all_span_idx_list_np.len: {dev_all_span_idx_list_np.shape}")

    # Test
    test_np = [npy_dict["input_ids"][dev_size:],
               npy_dict["attention_mask"][dev_size:],
               npy_dict["token_type_ids"][dev_size:]
              ]
    test_np = np.stack(test_np, axis=-1)
    test_label_ids_np = npy_dict["label_ids"][dev_size:]

    test_span_only_label_token_np = npy_dict["span_only_label_token"][dev_size:]
    test_all_span_len_list_np = npy_dict["all_span_len_list"][dev_size:]
    test_real_span_mask_token_np = npy_dict["real_span_mask_token"][dev_size:]
    test_all_span_idx_list_np = npy_dict["all_span_idx_list"][dev_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_label_ids.shape: {test_label_ids_np.shape}")

    print(f"test_span_only_label_token_np.shape: {test_span_only_label_token_np.shape}")
    print(f"test_all_span_len_list_np.len: {test_all_span_len_list_np.shape}")
    print(f"test_real_span_mask_token_np.len: {test_real_span_mask_token_np.shape}")
    print(f"test_all_span_idx_list_np.len: {test_all_span_idx_list_np.shape}")

    # Save
    root_path = "../corpus/npy/" + save_path
    # save input_ids, attention_mask, token_type_ids
    np.save(root_path + "/train", train_np)
    np.save(root_path + "/dev", dev_np)
    np.save(root_path + "/test", test_np)

    np.save(root_path + "/train_span_only_label_token", train_span_only_label_token_np)
    np.save(root_path + "/dev_span_only_label_token", dev_span_only_label_token_np)
    np.save(root_path + "/test_span_only_label_token", test_span_only_label_token_np)

    np.save(root_path + "/train_all_span_len_list", train_all_span_len_list_np)
    np.save(root_path + "/dev_all_span_len_list", dev_all_span_len_list_np)
    np.save(root_path + "/test_all_span_len_list", test_all_span_len_list_np)

    np.save(root_path + "/train_label_ids", train_label_ids_np)
    np.save(root_path + "/dev_label_ids", dev_label_ids_np)
    np.save(root_path + "/test_label_ids", test_label_ids_np)

    np.save(root_path + "/train_real_span_mask_token", train_real_span_mask_token_np)
    np.save(root_path + "/dev_real_span_mask_token", dev_real_span_mask_token_np)
    np.save(root_path + "/test_real_span_mask_token", test_real_span_mask_token_np)

    np.save(root_path + "/train_all_span_idx", train_all_span_idx_list_np)
    np.save(root_path + "/dev_all_span_idx", dev_all_span_idx_list_np)
    np.save(root_path + "/test_all_span_idx", test_all_span_idx_list_np)

    print("save complete")

### MAIN ###
if "__main__" == __name__:
    # load corpus
    pkl_src_path = "../corpus/pkl/NIKL_ne_pos.pkl"
    all_sent_list = []
    all_sent_list = load_ne_entity_list(src_path=pkl_src_path)

    make_span_npy(
        tokenizer_name="monologg/koelectra-base-v3-discriminator",
        src_list=all_sent_list, seq_max_len=128, span_max_len=4,
        debug_mode=False, save_npy_path="span_ner"
    )