import random
import re
import numpy as np

from pathlib import Path
from klue_tag_def import KLUE_NER_TAG, NerExample
from utils.tag_def import ETRI_TAG, MECAB_POS_TAG
from typing import List, Tuple

from transformers import ElectraTokenizer
from eunjeon import Mecab
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

#===========================================================
def create_ner_examples(src_path: str, tokenizer):
#===========================================================
    print(f"[create_ner_examples] src_path: {src_path}")

    all_sent_ne_pairs = []

    file_path = Path(src_path)
    raw_text = file_path.read_text(encoding="utf-8").strip()
    raw_docs = re.split(r"\n\t?\n", raw_text)
    for doc in raw_docs:
        original_clean_tokens = []  # clean tokens (bert clean func)
        original_clean_labels = []  # clean labels (bert clean func)

        sentence = ""
        for line in doc.split("\n"):
            if line[:2] == "##":
                guid = line.split("\t")[0].replace("##", "")
                continue
            token, tag = line.split("\t")
            sentence += token
            original_clean_tokens.append(token)
            original_clean_labels.append(tag)

        ne_word_label_pair = []  # [ word, label ]
        curr_word = ""
        curr_label = ""
        for char_tok, char_label in zip(original_clean_tokens, original_clean_labels):
            if "B-" in char_label:
                if 0 < len(curr_word) and 0 < len(curr_label):
                    ne_word_label_pair.append([curr_word, curr_label])
                curr_word = char_tok
                curr_label = char_label.replace("B-", "")
            elif "O" in char_label and "I-" not in char_label:
                if 0 < len(curr_word) and 0 < len(curr_label):
                    ne_word_label_pair.append([curr_word, curr_label])
                curr_word = ""
                curr_label = ""
            else:
                curr_word += char_tok
        if 0 < len(curr_word) and 0 < len(curr_label):
            ne_word_label_pair.append([curr_word, curr_label])

        all_sent_ne_pairs.append([sentence, ne_word_label_pair])

    return all_sent_ne_pairs

#===========================================================
def create_features(examples, tokenizer, target_n_pos, target_tag_list,
                    max_seq_len: int = 128, max_span_len: int = 8):
#===========================================================
    print(f"[create_features] Label list: {KLUE_NER_TAG}")

    span_minus = int((max_span_len + 1) * max_span_len / 2)
    max_num_span = int(max_seq_len * max_span_len - span_minus)

    mecab = Mecab()
    label_map = {label: i for i, label in enumerate(KLUE_NER_TAG)}
    '''
        klue_tag: 
            "B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG",
            "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT",
    '''
    etri_tags = {'O': 0, 'FD': 1, 'EV': 2, 'DT': 3, 'TI': 4, 'MT': 5,
                 'AM': 6, 'LC': 7, 'CV': 8, 'PS': 9, 'TR': 10,
                 'TM': 11, 'AF': 12, 'PT': 13, 'OG': 14, 'QT': 15}
    ne_ids2tag = {v: i for i, v in enumerate(etri_tags)}
    ne_detail_ids2_tok = {v: k for k, v in ETRI_TAG.items()}

    random.seed(42)
    random.shuffle(examples)

    npy_dict = {
        "input_ids": [],
        "label_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "all_span_len_list": [],
        "real_span_mask_token": [],
        "span_only_label_token": [],
        "all_span_idx_list": [],

        "pos_ids": []
    }
    '''
        example
            [0] sentence, [1] : [[0] ne_word, [1] ne_tag]
    '''
    for ex_idx, example in enumerate(examples):
        sentence = example[0]
        ex_ne_list = example[1]

        if 0 == (ex_idx % 1000):
            print(f"{ex_idx} Processing... {example[0]}")

        mecab_res = mecab.pos(sentence, False)
        # [('전창수', 'NNP', False), ('(', 'SSO', False), ('42', 'SN', False)]
        conv_mecab_res = convert_morp_connected_tokens(mecab_res)

        origin_tokens = []
        text_tokens = []
        token_pos_list = []
        for m_idx, mecab_item in enumerate(conv_mecab_res):
            tokens = tokenizer.tokenize(mecab_item[0])
            origin_tokens.extend(tokens)

            if mecab_item[-1]:
                for tok in tokens:
                    text_tokens.append("##" + tok)
            else:
                text_tokens.extend(tokens)

            # 0 index에는 [CLS] 토큰이 있어야 한다.
            for _ in range(len(tokens)):
                token_pos_list.append(mecab_item[1].split("+"))

        origin_token_ids = tokenizer.convert_tokens_to_ids(origin_tokens)
        conv_token_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        for tok_idx, (ori_tok, conv_tok) in enumerate(zip(origin_token_ids, conv_token_ids)):
            if 1 == conv_tok and 1 != ori_tok:
                # print(text_tokens[tok_idx], origin_tokens[tok_idx])
                text_tokens[tok_idx] = origin_tokens[tok_idx]

        # morp_ids
        mecab_type_len = len(MECAB_POS_TAG.keys())
        mecab_tag2id = {v: k for k, v in MECAB_POS_TAG.items()}
        pos_ids = [[]] # [CLS]
        for tok_pos in token_pos_list:
            curr_pos = []
            # curr_add_idx = 1
            for pos in tok_pos:
                filter_pos = pos if "UNKNOWN" != pos and "NA" != pos and "UNA" != pos and "VSV" != pos else "O"
                # pos_idx = mecab_tag2id[filter_pos]
                curr_pos.append(mecab_tag2id[filter_pos])
                # curr_add_idx += 1
            pos_ids.append(curr_pos)

        if max_seq_len <= len(pos_ids):
            pos_ids = pos_ids[:max_seq_len - 1]
            pos_ids.append([0 for _ in range(mecab_type_len)]) # [SEP]
        else:
            pos_ids_size = len(pos_ids)
            for _ in range(max_seq_len - pos_ids_size):
                pos_ids.append([0 for _ in range(mecab_type_len)])

        # Text Tokens
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        if max_seq_len <= len(text_tokens):
            text_tokens = text_tokens[:max_seq_len - 1]
            text_tokens.append("[SEP]")

            valid_token_len = max_seq_len
        else:
            text_tokens.append("[SEP]")
            valid_token_len = len(text_tokens)

        # NE - Token 단위
        label_ids = [ETRI_TAG["O"]] * len(text_tokens)
        b_check_use = [False for _ in range(len(text_tokens))]
        for ne_idx, ne_item in enumerate(ex_ne_list):
            ne_char_list = list(ne_item[0].replace(" ", ""))
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
                    label_ids[bio_idx] = ETRI_TAG["B-" + ne_item[1]]
                else:
                    label_ids[bio_idx] = ETRI_TAG["I-" + ne_item[1]]

        if max_seq_len <= len(label_ids):
            label_ids = label_ids[:max_seq_len - 1]
            label_ids.append(ETRI_TAG["O"])
        else:
            label_ids_size = len(label_ids)
            for _ in range(max_seq_len - label_ids_size):
                label_ids.append(ETRI_TAG["O"])

        all_span_idx_list = enumerate_spans(text_tokens, offset=0, max_span_width=max_span_len)
        all_span_len_list = []
        for idx_pair in all_span_idx_list:
            s_idx, e_idx = idx_pair
            span_len = e_idx - s_idx + 1
            all_span_len_list.append(span_len)

        span_idx_label_dict = make_span_idx_label_pair(ex_ne_list, text_tokens)
        span_idx_new_label_dict = convert2tokenIdx(span_idxLab=span_idx_label_dict,
                                                   all_span_idxs=all_span_idx_list)
        span_only_label_token = []  # 만들어진 span 집합들의 label
        for idx_str, label in span_idx_new_label_dict.items():
            span_only_label_token.append(ne_ids2tag[label])

        text_tokens += ["[PAD]"] * (max_seq_len - valid_token_len)
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = ([1] * valid_token_len) + ([0] * (max_seq_len - valid_token_len))
        token_type_ids = [0] * max_seq_len

        # Span Len
        all_span_idx_list = all_span_idx_list[:max_num_span]
        span_only_label_token = span_only_label_token[:max_num_span]
        all_span_len_list = all_span_len_list[:max_num_span]
        real_span_mask_token = np.ones_like(span_only_label_token).tolist()

        mecab_tag2ids = {v: k for k, v in MECAB_POS_TAG.items()}  # origin, 1: "NNG"
        # {1: 0, 2: 1, 43: 2, 3: 3, 5: 4, 4: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, 23: 11, 24: 12, 21: 13, 22: 14}
        target_tag2ids = {mecab_tag2ids[x]: i for i, x in enumerate(target_tag_list)}
        pos_target_onehot = []
        for start_idx, end_idx in all_span_idx_list:
            span_pos = [0 for _ in range(target_n_pos)]
            for pos in pos_ids[start_idx:end_idx + 1]:
                for pos_item in pos:  # @TODO: Plz Check
                    if pos_item in target_tag2ids.keys():
                        if 0 == pos_item or 1 == pos_item:
                            span_pos[0] = 1
                        else:
                            span_pos[target_tag2ids[pos_item] - 1] = 1
            pos_target_onehot.append(span_pos)

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
        if max_num_span > len(pos_target_onehot):
            diff_len = max_num_span - len(pos_target_onehot)
            pos_target_onehot += [[0 for _ in range(target_n_pos)]] * diff_len

        assert len(input_ids) == max_seq_len, f"{len(input_ids)}"
        assert len(attention_mask) == max_seq_len, f"{len(attention_mask)}"
        assert len(token_type_ids) == max_seq_len, f"{len(token_type_ids)}"
        assert len(label_ids) == max_seq_len, f"{len(label_ids)}"
        assert len(pos_ids) == max_seq_len, f"{len(pos_ids)}"

        assert len(span_only_label_token) == max_num_span, f"{len(span_only_label_token)}"
        assert len(all_span_idx_list) == max_num_span, f"{len(all_span_idx_list)}"
        assert len(all_span_len_list) == max_num_span, f"{len(all_span_len_list)}"
        assert len(real_span_mask_token) == max_num_span, f"{len(real_span_mask_token)}"
        assert len(pos_target_onehot) == max_num_span, f"{len(pos_target_onehot)}"

        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["label_ids"].append(label_ids)

        npy_dict["span_only_label_token"].append(span_only_label_token)
        npy_dict["all_span_len_list"].append(all_span_len_list)
        npy_dict["real_span_mask_token"].append(real_span_mask_token)
        npy_dict["all_span_idx_list"].append(all_span_idx_list)

        npy_dict["pos_ids"].append(pos_target_onehot)

    # Save npy
    save_span_npy(npy_dict, save_path="../corpus/npy/klue_ner")

#=======================================================================================
def save_span_npy(npy_dict, save_path):
#=======================================================================================
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["label_ids"] = np.array(npy_dict["label_ids"])

    npy_dict["span_only_label_token"] = np.array(npy_dict["span_only_label_token"])
    npy_dict["all_span_len_list"] = np.array(npy_dict["all_span_len_list"])
    npy_dict["real_span_mask_token"] = np.array(npy_dict["real_span_mask_token"])
    npy_dict["all_span_idx_list"] = np.array(npy_dict["all_span_idx_list"])

    npy_dict["pos_ids"] = np.array(npy_dict["pos_ids"])

    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"label_ids.shape: {npy_dict['label_ids'].shape}")

    print(f"span_only_label_token.shape: {npy_dict['span_only_label_token'].shape}")
    print(f"all_span_len_list.shape: {npy_dict['all_span_len_list'].shape}")
    print(f"real_span_mask_token.shape: {npy_dict['real_span_mask_token'].shape}")
    print(f"all_span_idx_list.shape: {npy_dict['all_span_idx_list'].shape}")

    print(f"pos_ids.shape: {npy_dict['pos_ids'].shape}")

    # Train
    train_np = [npy_dict["input_ids"],
                npy_dict["attention_mask"],
                npy_dict["token_type_ids"]]
    train_np = np.stack(train_np, axis=-1)
    train_label_ids_np = npy_dict["label_ids"]

    train_span_only_label_token_np = npy_dict["span_only_label_token"]
    train_all_span_len_list_np = npy_dict["all_span_len_list"]
    train_real_span_mask_token_np = npy_dict["real_span_mask_token"]
    train_all_span_idx_list_np = npy_dict["all_span_idx_list"]

    train_pos_ids_np = npy_dict["pos_ids"]

    # Save
    root_path = save_path
    # save input_ids, attention_mask, token_type_ids
    np.save(root_path + "/train", train_np)

    np.save(root_path + "/train_span_only_label_token", train_span_only_label_token_np)
    np.save(root_path + "/train_all_span_len_list", train_all_span_len_list_np)
    np.save(root_path + "/train_label_ids", train_label_ids_np)
    np.save(root_path + "/train_real_span_mask_token", train_real_span_mask_token_np)
    np.save(root_path + "/train_all_span_idx", train_all_span_idx_list_np)
    np.save(root_path + "/train_pos_ids", train_pos_ids_np)

    print("[save_span_npy] save complete")

#===========================================================
def create_npy_datasets(src_path: str, target_n_pos: int, target_tag_list: List):
#===========================================================
    print(f"[create_npy_datasets] src_path: {src_path}")

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    examples = create_ner_examples(src_path, tokenizer)
    create_features(examples, tokenizer, target_n_pos, target_tag_list, 128)

#=======================================================================================
def convert_morp_connected_tokens(sent_lvl_pos: Tuple[str, str]):
#=======================================================================================
    ret_conv_morp_tokens = []

    g_SYMBOL_TAGS = [
        "SF", "SE", # (마침표, 물음표, 느낌표), 줄임표
        "SSO", "SSC", # 여는 괄호, 닫는 괄호
        "SC", "SY", # 구분자, (붙임표, 기타 기호)
    ]

    for eojeol in sent_lvl_pos:
        for mp_idx, morp in enumerate(eojeol): # morp (마케팅, NNG)
            if 0 == mp_idx or morp[1] in g_SYMBOL_TAGS:
                ret_conv_morp_tokens.append((morp[0], morp[1], False))
            else:
                if eojeol[mp_idx-1][1] not in g_SYMBOL_TAGS:
                    conv_morp = (morp[0], morp[1], True)
                    ret_conv_morp_tokens.append(conv_morp)
                else:
                    ret_conv_morp_tokens.append((morp[0], morp[1], False))

    return ret_conv_morp_tokens

#=======================================================================================
def make_span_idx_label_pair(ne_list, text_tokens):
#=======================================================================================
    ret_dict = {}

    # print(text_tokens)
    b_check_use = [False for _ in range(len(text_tokens))]
    for ne_idx, ne_item in enumerate(ne_list):
        ne_char_list = list(ne_item[0].replace(" ", ""))

        concat_item_list = []
        for tok_idx in range(len(text_tokens)):
            if b_check_use[tok_idx]:
                continue
            for sub_idx in range(tok_idx + 1, len(text_tokens)):
                concat_word = ["".join(x).replace("##", "") for x in text_tokens[tok_idx:sub_idx]]
                concat_item_list.append(("".join(concat_word), (tok_idx, sub_idx - 1))) # Modify -1
        concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
        concat_item_list.sort(key=lambda x: len(x[0]))
        # print(ne_item.text, ne_item.type, concat_item_list)
        if 0 >= len(concat_item_list):
            continue
        target_idx_pair = concat_item_list[0][1]
        for bio_idx in range(target_idx_pair[0], target_idx_pair[1] + 1):
            b_check_use[bio_idx] = True
        key = str(target_idx_pair[0]) + ";" + str(target_idx_pair[1])
        ret_dict[key] = ne_item[1]

    # print(ret_dict)
    return ret_dict

#=======================================================================================
def convert2tokenIdx(all_span_idxs, span_idxLab):
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

### MAIN ###
if "__main__" == __name__:
    print("[klue_parser] __MAIN__ !")

    train_data_path = "./data/klue-ner-v1.1_train.tsv"
    dev_data_path = "./data/klue-ner-v1.1_dev.tsv"

    target_n_pos = 14
    target_tag_list = [  # NN은 NNG/NNP 통합
        "NNG", "NNP", "SN", "NNB", "NR", "NNBC",
        "JKS", "JKC", "JKG", "JKO", "JKB", "JX", "JC", "JKV", "JKQ",
    ]
    create_npy_datasets(src_path=dev_data_path, target_n_pos=target_n_pos, target_tag_list=target_tag_list)
    create_npy_datasets(src_path=train_data_path, target_n_pos=target_n_pos, target_tag_list=target_tag_list)