import re
import copy
import random
import numpy as np
import pickle

from tag_def import ETRI_TAG, LS_Tag, NIKL_POS_TAG, MECAB_POS_TAG

from typing import List, Dict, Tuple
from data_def import Sentence, NE, Morp

from transformers import AutoTokenizer, ElectraTokenizer
from dict_maker import Dict_Item, make_dict_hash_table

### global
random.seed(42)
np.random.seed(42)

#==============================================================
def conv_NIKL_pos_giho_category(sent_list: List[Sentence], is_status_nn: bool=False, is_verb_nn: bool=False):
#==============================================================
    if is_status_nn:
        with open("../corpus/nn_list/nn_sang_list.pkl", mode="rb") as status_nn_pkl:
            status_nn_list = pickle.load(status_nn_pkl)
            print(f"[conv_NIKL_pos_giho_categoy] status_nn_list.len: {len(status_nn_list)}")
    if is_verb_nn:
        with open("../corpus/nn_list/nn_verb_list.pkl", mode="rb") as verb_nn_pkl:
            verb_nn_list = pickle.load(verb_nn_pkl)
            print(f"[conv_NIKL_pos_giho_categoy] verb_nn_list.len: {len(verb_nn_list)}")

    '''
    SF, SW not converted
    쉼표, 가운뎃점, 콜론, 빗금 (SP), 따옴표, 괄호표, 줄표(SS), 줄임표(SE), 붙임표(물결) (SO) -> SP
    '''
    for sent_item in sent_list:
        for morp_item in sent_item.morp_list:
            if "SS" == morp_item.label or "SE" == morp_item.label or "SO" == morp_item.label:
                morp_item.label = "SP"
            if "NNG" == morp_item.label or "NNP" == morp_item.label or "NNB" == morp_item.label:
                if is_status_nn and morp_item.form in status_nn_list:
                    morp_item.label = "STATUS_NN"
                    # print("\nAAAAAAAAAA:\n", morp_item)
                    # input()
                elif is_verb_nn and morp_item.form in verb_nn_list:
                    morp_item.label = "VERB_NN"
                    # print("\nBBBBBBBBB:\n", morp_item)
                    # input()

    return sent_list

#==============================================================
def conv_TTA_ne_category(sent_list: List[Sentence]):
#==============================================================
    '''
    NE Type : 세분류 -> 대분류

    Args:
        sent_list: list of Sentence dataclass
    '''
    for sent_item in sent_list:
        for ne_item in sent_item.ne_list:
            conv_type = ""
            lhs_type = ne_item.type.split("_")[0]
            if "PS" in lhs_type:
                conv_type = "PS"
            elif "LC" in lhs_type:
                conv_type = "LC"
            elif "OG" in lhs_type:
                conv_type = "OG"
            elif "AF" in lhs_type:
                conv_type = "AF"
            elif "DT" in lhs_type:
                conv_type = "DT"
            elif "TI" in lhs_type:
                conv_type = "TI"
            elif "CV" in lhs_type:
                conv_type = "CV"
            elif "AM" in lhs_type:
                conv_type = "AM"
            elif "PT" in lhs_type:
                conv_type = "PT"
            elif "QT" in lhs_type:
                conv_type = "QT"
            elif "FD" in lhs_type:
                conv_type = "FD"
            elif "TR" in lhs_type:
                conv_type = "TR"
            elif "EV" in lhs_type:
                conv_type = "EV"
            elif "MT" in lhs_type:
                conv_type = "MT"
            elif "TM" in lhs_type:
                conv_type = "TM"
            else:
                print(f"What is {lhs_type}")
                return

            if "" == conv_type:
                print(sent_item.text, "\n", lhs_type)
                return
            ne_item.type = conv_type
    return sent_list

#==============================================================
def save_npy_dict(npy_dict: Dict[str, List], src_list_len, save_model_dir):
#==============================================================
    # convert list to numpy
    print(f"[save_npy_dict] Keys - {npy_dict.keys()}")

    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["token_seq_len"] = np.array(npy_dict["token_seq_len"])
    npy_dict["eojeol_ids"] = np.array(npy_dict["eojeol_ids"])
    # npy_dict["entity_ids"] = np.array(npy_dict["entity_ids"])
    npy_dict["pos_tag_ids"] = np.array(npy_dict["pos_tag_ids"])

    print(f"[save_npy_dict] input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"[save_npy_dict] attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"[save_npy_dict] token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"[save_npy_dict] labels.shape: {npy_dict['labels'].shape}")
    print(f"[save_npy_dict] token_seq_len.shape: {npy_dict['token_seq_len'].shape}")
    print(f"[save_npy_dict] eojeol_ids.shape: {npy_dict['eojeol_ids'].shape}")
    # print(f"[save_npy_dict] entity_ids.shape: {npy_dict['entity_ids'].shape}")
    print(f"[save_npy_dict] pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")

    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    # Train
    train_np = [
        npy_dict["input_ids"][:train_size],
        npy_dict["attention_mask"][:train_size],
        npy_dict["token_type_ids"][:train_size]
    ]
    train_np = np.stack(train_np, axis=-1)
    train_labels = npy_dict["labels"][:train_size]
    train_token_seq_len_np = npy_dict["token_seq_len"][:train_size]
    train_eojeol_ids_np = npy_dict["eojeol_ids"][:train_size]
    # train_entity_ids_np = npy_dict["entity_ids"][:train_size]
    train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]

    print(f"[save_npy_dict] train_np.shape: {train_np.shape}")
    print(f"[save_npy_dict] train_labels.shape: {train_labels.shape}")
    print(f"[save_npy_dict] train_token_seq_len_np.shape: {train_token_seq_len_np.shape}")
    print(f"[save_npy_dict] train_eojeol_ids_np.shape: {train_eojeol_ids_np.shape}")
    # print(f"[save_npy_dict] train_entity_ids_np.shape: {train_entity_ids_np.shape}")
    print(f"[save_npy_dict] train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")

    # Dev
    dev_np = [
        npy_dict["input_ids"][train_size:valid_size],
        npy_dict["attention_mask"][train_size:valid_size],
        npy_dict["token_type_ids"][train_size:valid_size]
    ]
    dev_np = np.stack(dev_np, axis=-1)
    dev_labels_np = npy_dict["labels"][train_size:valid_size]
    dev_token_seq_len_np = npy_dict["token_seq_len"][train_size:valid_size]
    dev_eojeol_ids_np = npy_dict["eojeol_ids"][train_size:valid_size]
    # dev_entity_ids_np = npy_dict["entity_ids"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]

    print(f"[save_npy_dict] dev_np.shape: {dev_np.shape}")
    print(f"[save_npy_dict] dev_labels_np.shape: {dev_labels_np.shape}")
    print(f"[save_npy_dict] dev_token_seq_len_np.shape: {dev_token_seq_len_np.shape}")
    print(f"[save_npy_dict] dev_eojeol_ids_np.shape: {dev_eojeol_ids_np.shape}")
    # print(f"[save_npy_dict] dev_entity_ids_np.shape: {dev_entity_ids_np.shape}")
    print(f"[save_npy_dict] dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")

    # Test
    test_np = [
        npy_dict["input_ids"][valid_size:],
        npy_dict["attention_mask"][valid_size:],
        npy_dict["token_type_ids"][valid_size:]
    ]
    test_np = np.stack(test_np, axis=-1)
    test_labels_np = npy_dict["labels"][valid_size:]
    test_token_seq_len_np = npy_dict["token_seq_len"][valid_size:]
    test_eojeol_ids_np = npy_dict["eojeol_ids"][valid_size:]
    # test_entity_ids_np = npy_dict["entity_ids"][valid_size:]
    test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]

    print(f"[save_npy_dict] test_np.shape: {test_np.shape}")
    print(f"[save_npy_dict] test_labels_np.shape: {test_labels_np.shape}")
    print(f"[save_npy_dict] test_token_seq_len_np.shape: {test_token_seq_len_np.shape}")
    print(f"[save_npy_dict] test_eojeol_ids_np.shape: {test_eojeol_ids_np.shape}")
    # print(f"[save_npy_dict] test_entity_ids_np.shape: {test_entity_ids_np.shape}")
    print(f"[save_npy_dict] test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")

    # save input_ids, attention_mask, token_type_ids
    root_path = "../corpus/npy/"+save_model_dir
    print(f"[save_npy_dict] root_path : {root_path}")

    np.save(root_path + "/train", train_np)
    np.save(root_path + "/dev", dev_np)
    np.save(root_path + "/test", test_np)

    # save labels
    np.save(root_path + "/train_labels", train_labels)
    np.save(root_path + "/dev_labels", dev_labels_np)
    np.save(root_path + "/test_labels", test_labels_np)

    # save token_seq_len
    np.save(root_path + "/train_token_seq_len", train_token_seq_len_np)
    np.save(root_path + "/dev_token_seq_len", dev_token_seq_len_np)
    np.save(root_path + "/test_token_seq_len", test_token_seq_len_np)

    # save eojeol_ids
    np.save(root_path + "/train_eojeol_ids", train_eojeol_ids_np)
    np.save(root_path + "/dev_eojeol_ids", dev_eojeol_ids_np)
    np.save(root_path + "/test_eojeol_ids", test_eojeol_ids_np)

    # save entity_ids
    # np.save("../data/npy/old_nikl/electra/train_entity_ids", train_entity_ids_np)
    # np.save("../data/npy/old_nikl/electra/dev_entity_ids", dev_entity_ids_np)
    # np.save("../data/npy/old_nikl/electra/test_entity_ids", test_entity_ids_np)

    # save pos_tag
    np.save(root_path + "/train_pos_tag", train_pos_tag_np)
    np.save(root_path + "/dev_pos_tag", dev_pos_tag_np)
    np.save(root_path + "/test_pos_tag", test_pos_tag_np)

#==============================================================
def make_wordpiece_npy(
        tokenizer_name: str, src_list: List[Sentence], max_pos_nums: int=4,
        max_len: int=128, debug_mode: bool=False,
        save_model_dir: str = None
):
#==============================================================
    if not save_model_dir:
        print(f"[make_wordpiece_npy] Plz check save_model_dir: {save_model_dir}")
        return

    random.shuffle(src_list)

    npy_dict = {
        "input_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "labels": [],
        "token_seq_len": [],
        "eojeol_ids": [],
        "pos_tag_ids": [],
    }
    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in NIKL_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}

    if "bert" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    for proc_idx, src_item in enumerate(src_list):
        # Test
        # if "29·미국·사진" not in src_item.text:
        #     continue
        # if "전창수(42) 두산타워 마케팅팀 차장" not in src_item.text:
        #     continue
        # if '샌드위치→역(逆)샌드위치→신(新)샌드위치….' not in src_item.text:
        #     continue
        # if "그동안 각 언론사와 후보 진영이 실시한 여론조사에서도 홍준표·원희룡·나경원 후보가 '3강'을 형성하며 엎치락뒤치락해 왔다." not in src_item.text:
        #     continue
        # if "P 불투르(Vulture) 인사위원회 위원장은" not in src_item.text:
        #     continue
        # if "넙치·굴비·홍어·톳·꼬시래기·굴·홍합" not in src_item.text:
        #     continue
        # if "LG 우규민-삼성 웹스터(대구)" not in src_item.text:
        #     continue

        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")
        text_tokens = tokenizer.tokenize(src_item.text)

        # make (word, token, pos) pair
        # [(word, [tokens], (begin, end))]
        word_tokens_pos_pair_list: List[Tuple[str, List[str], Tuple[int, int]]] = []
        for word_idx, word_item in enumerate(src_item.word_list):
            form_tokens = tokenizer.tokenize(word_item.form)
            word_tokens_pos_pair_list.append((word_item.form, form_tokens, (word_item.begin, word_item.end)))

        labels = ["O"] * len(text_tokens)
        eojeol_ids = [0] * len(text_tokens)
        pos_tag_ids = []
        for _ in range(len(text_tokens)):
            pos_tag_ids.append(["O"] * max_pos_nums)

        # NE
        start_idx = 0
        for ne_item in src_item.ne_list:
            is_find = False
            for s_idx in range(start_idx, len(text_tokens)):
                if is_find:
                    break
                for word_cnt in range(0, len(text_tokens) - s_idx + 1):
                    concat_text_tokens = text_tokens[s_idx:s_idx + word_cnt]
                    concat_text_tokens = [x.replace("##", "") for x in concat_text_tokens]

                    if "".join(concat_text_tokens) == ne_item.text.replace(" ", ""):
                        # BIO Tagging
                        for bio_idx in range(s_idx, s_idx + word_cnt):
                            if bio_idx == s_idx:
                                labels[bio_idx] = "B-" + ne_item.type
                            else:
                                labels[bio_idx] = "I-" + ne_item.type

                        is_find = True
                        start_idx = s_idx + word_cnt
                        break
        ## end, NE loop

        # Morp
        start_idx = 0
        for morp_item in src_item.morp_list:
            is_find = False
            split_morp_label_item = morp_item.label.split("+")

            for s_idx in range(start_idx, len(text_tokens)):
                if is_find:
                    break
                for word_size in range(0, len(text_tokens) - s_idx + 1):
                    concat_text_tokens = text_tokens[s_idx:s_idx + word_size]
                    concat_text_tokens = [x.replace("##", "") for x in concat_text_tokens]
                    concat_str = "".join(concat_text_tokens)
                    if len(concat_str) > len(morp_item.form):
                        break
                    if pos_tag_ids[s_idx][0] != "O":
                        continue
                    if concat_str == morp_item.form:
                        for sp_idx, sp_item in enumerate(split_morp_label_item):
                            if 2 < sp_idx:
                                break
                            pos_tag_ids[s_idx][sp_idx] = sp_item
                            if "" == sp_item:  # Exception
                                pos_tag_ids[s_idx][sp_idx] = "O"
                        is_find = True
                        break
        # end, Morp Loop

        # Eojeol Loop
        add_idx = 0
        for eojeol_idx, wtp_item in enumerate(word_tokens_pos_pair_list):
            wtp_token_size = len(wtp_item[1])
            for _ in range(wtp_token_size):
                eojeol_ids[add_idx] = eojeol_idx + 1
                add_idx += 1
        # end, Eojeol_ids Loop

        text_tokens.insert(0, "[CLS]")
        labels.insert(0, "O")
        pos_tag_ids.insert(0, (["O"] * max_pos_nums))
        eojeol_ids.insert(0, 0)

        valid_len = 0
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len - 1]
            labels = labels[:max_len - 1]
            pos_tag_ids = pos_tag_ids[:max_len - 1]
            eojeol_ids = eojeol_ids[:max_len - 1]

            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * max_pos_nums))
            eojeol_ids.append(0)

            valid_len = max_len
        else:
            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * max_pos_nums))
            eojeol_ids.append(0)

            valid_len = len(text_tokens)
            text_tokens = text_tokens + ["[PAD]"] * (max_len - valid_len)
            labels = labels + ["O"] * (max_len - valid_len)
            eojeol_ids = eojeol_ids + [0] * (max_len - valid_len)
            for _ in range(max_len - valid_len):
                pos_tag_ids.append((["O"] * max_pos_nums))

        attention_mask = ([1] * valid_len) + ([0] * (max_len - valid_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        labels = [ETRI_TAG[x] for x in labels]
        for i in range(max_len):
            pos_tag_ids[i] = [pos_tag2ids[x] for x in pos_tag_ids[i]]

        assert len(input_ids) == max_len, f"{input_ids} + {len(input_ids)}"
        assert len(labels) == max_len, f"{labels} + {len(labels)}"
        assert len(attention_mask) == max_len, f"{attention_mask} + {len(attention_mask)}"
        assert len(token_type_ids) == max_len, f"{token_type_ids} + {len(token_type_ids)}"
        assert len(pos_tag_ids) == max_len, f"{pos_tag_ids} + {len(pos_tag_ids)}"
        assert len(eojeol_ids) == max_len, f"{eojeol_ids} + {len(eojeol_ids)}"

        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["labels"].append(labels)

        # token_seq_len
        npy_dict["token_seq_len"].append(valid_len)

        # eojeol_ids
        npy_dict["eojeol_ids"].append(eojeol_ids)

        # pos_tag_ids
        pos_tag_ids = convert_pos_tag_to_combi_tag(pos_tag_ids)
        npy_dict["pos_tag_ids"].append(pos_tag_ids)

        if debug_mode:
            print(f"Sent: ")
            print("Origin Sent Tokens: ", tokenizer.tokenize(src_item.text))
            print("Eojeol Sent Tokens: ", text_tokens)
            print(word_tokens_pos_pair_list)
            print(f"Sequence length: {valid_len}\n")
            print(f"NE: {src_item.ne_list}\n")
            print(f"Morp: {src_item.morp_list}\n")
            conv_pos_tag_ids = [[pos_ids2tag[x] for x in pos_tag_item] for pos_tag_item in pos_tag_ids]
            for ii, am, tti, label, pti, ej in zip(input_ids, attention_mask, token_type_ids,
                                                   labels, conv_pos_tag_ids, eojeol_ids):
                print(ii, tokenizer.convert_ids_to_tokens([ii]), ne_ids2tag[label], am, tti, pti, ej)
            input()

    # save
    save_npy_dict(npy_dict, src_list_len=len(src_list), save_model_dir=save_model_dir)

#==============================================================
def make_pos_tag_npy(tokenizer_name: str, src_list: List[Sentence], max_len: int=512):
#==============================================================
    '''
    Note:
        Sentence + Named Entity + Morp Token Npy

        tokenization base is pos tagging
    '''

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "seq_len": [],
        "pos_tag_ids": [],
    }
    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for proc_idx, sent in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {sent.text}")

        text_tokens = []
        for mp_item in sent.morp_list:
            mp_tokens = tokenizer.tokenize(mp_item.form)
            text_tokens.extend(mp_tokens)
        labels = ["O"] * len(text_tokens)
        pos_tag_ids = []
        for _ in range(len(text_tokens)):
            pos_tag_ids.append(["O"] * 3)

        # NE
        start_idx = 0
        for ne_item in sent.ne_list:
            is_find = False
            for s_idx in range(start_idx, len(text_tokens)):
                if is_find:
                    break
                for word_cnt in range(0, len(text_tokens) - s_idx + 1):
                    concat_text_tokens = text_tokens[s_idx:s_idx+word_cnt]
                    concat_text_tokens = [x.replace("##", "") for x in concat_text_tokens]

                    if "".join(concat_text_tokens) == ne_item.text.replace(" ", ""):
                        # print("A : ", concat_text_tokens, ne_item.text)

                        # BIO Tagging
                        for bio_idx in range(s_idx, s_idx+word_cnt):
                            if bio_idx == s_idx:
                                labels[bio_idx] = "B-" + ne_item.type
                            else:
                                labels[bio_idx] = "I-" + ne_item.type

                        is_find = True
                        start_idx = s_idx
                        break
        ## end, ne_item loop

        # TEST and Print
        # test_ne_print = [(x.text, x.type) for x in sent.ne_list]
        # id2la = {v: k for k, v in ETRI_TAG.items()}
        # print(test_ne_print)
        # for t, l in zip(text_tokens, labels):
        #     print(t, "\t", l)

        # Morp
        pos_idx = 0
        for morp_item in sent.morp_list:
            mp_tokens = tokenizer.tokenize(morp_item.form)

            split_morp_label_item = morp_item.label.split("+")
            for _ in range(len(mp_tokens)):
                for sp_idx, split_label in enumerate(split_morp_label_item):
                    if 2 < sp_idx:
                        continue
                    pos_tag_ids[pos_idx][sp_idx] = split_label
                    if "" == split_label:
                        pos_tag_ids[pos_idx][sp_idx] = "O"
                pos_idx += 1

        text_tokens.insert(0, "[CLS]")
        labels.insert(0, "O")
        pos_tag_ids.insert(0, (["O"] * 3))

        valid_len = 0
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len - 1]
            labels = labels[:max_len - 1]
            pos_tag_ids = pos_tag_ids[:max_len - 1]

            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * 3))

            valid_len = max_len
        else:
            text_tokens.append("[SEP]")
            labels.append("O")
            pos_tag_ids.append((["O"] * 3))

            valid_len = len(text_tokens)
            text_tokens = text_tokens + ["[PAD]"] * (max_len - valid_len)
            labels = labels + ["O"] * (max_len - valid_len)
            for _ in range(max_len - valid_len):
                pos_tag_ids.append((["O"] * 3))

        attention_mask = ([1] * valid_len) + ([0] * (max_len - valid_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        labels = [ETRI_TAG[x] for x in labels]
        for i in range(max_len):
            pos_tag_ids[i] = [pos_tag2ids[x] for x in pos_tag_ids[i]]

        assert len(input_ids) == max_len, f"{input_ids} + {len(input_ids)}"
        assert len(labels) == max_len, f"{labels} + {len(labels)}"
        assert len(attention_mask) == max_len, f"{attention_mask} + {len(attention_mask)}"
        assert len(token_type_ids) == max_len, f"{token_type_ids} + {len(token_type_ids)}"
        assert len(pos_tag_ids) == max_len, f"{pos_tag_ids} + {len(pos_tag_ids)}"

        npy_dict["input_ids"].append(input_ids)
        npy_dict["labels"].append(labels)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)

        # for pack_padded_sequence
        npy_dict["seq_len"].append(valid_len)

        # add pos tag info
        npy_dict["pos_tag_ids"].append(pos_tag_ids)

    # save
    save_npy_dict(npy_dict, src_list_len=len(src_list))
####

#==============================================================
def make_eojeol_datasets_npy(
        tokenizer_name: str, src_list: List[Sentence],
        max_len: int = 128, eojeol_max_len: int = 50, debug_mode: bool = False,
        save_model_dir: str = None
):
#==============================================================
    if not save_model_dir:
        print(f"[make_eojeol_datasets_npy] Plz check save_model_dir: {save_model_dir}")
        return

    random.shuffle(src_list)

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        #"token_seq_len": [],
        "pos_tag_ids": [],
        "eojeol_ids": [],
        "LS_ids": [],
        # "entity_ids": [], # for token_type_ids (bert segment embedding)
    }
    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in NIKL_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}
    ls_ids2tag = {v: k for k, v in LS_Tag.items()}

    if "bert" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    # Test sentences
    test_str_list = [
        "29·미국·사진", "전창수(42) 두산타워 마케팅팀 차장", "샌드위치→역(逆)샌드위치→신(新)샌드위치….",
        "홍준표·원희룡·나경원 후보가 '3강'을 형성하며 엎치락뒤치락해 왔다.", "P 불투르(Vulture) 인사위원회 위원장은",
        "넙치·굴비·홍어·톳·꼬시래기·굴·홍합", "연준 의장이", "황병서 북한군 총정치국장이 올해 10월 4일",
        "영업익 4482억 ‘깜짝’… LG전자 ‘부활의 노래’", "LG 우규민-삼성 웹스터(대구)",
        "‘김종영 그 절대를 향한’ 전시회", "재산증가액이 3억5000만원이다.", "‘진실·화해를 위한 과거사 정리위원회’",
        "용의자들은 25일 아침 9시께", "해외50여 개국에서 5500회 이상 공연하며 사물놀이",
        "REDD는 열대우림 등 산림자원을 보호하는 개도국이나",
        "2010년 12월부터 이미 가중 처벌을 시행하는 어린이 보호구역의 교통사고 발생 건수는",
        "금리설계형의 경우 변동금리(6개월 변동 코픽스 연동형)는", "현재 중국의 공항은 400여 개다.",
        "'중국 편'이라고 믿었던 박 대통령에게"
    ]
    test_str_list = [
        "'중국 편'이라고 믿었던 박 대통령에게"
    ]

    for proc_idx, src_item in enumerate(src_list):
        # Test
        if debug_mode:
            is_test_str = False
            for test_str in test_str_list:
                if test_str in src_item.text:
                    is_test_str = True
            if not is_test_str:
                continue

        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")

        # make (word, token, pos) pair
        # [(word, [tokens], [POS])]
        word_tokens_pos_pair_list: List[Tuple[str, List[str], List[str]]] = []
        # separation_giho = ["「", "」", "…", "〔", "〕", "(", ")",
        #                    "\"", "…", "...", "→", "_", "|", "〈", "〉",
        #                    "?", "!", "<", ">", "ㆍ", "•", "《", "》",
        #                    "[", "]", "ㅡ", "+", "“", "”", ";", "·",
        #                    "‘", "’", "″", "″", "'", "'", "-", "~", ",", "."]
        split_giho_label = [
            "SF", "SP", "SS", "SE", "SO", "SW"
        ]
        for word_idx, word_item in enumerate(src_item.word_list):
            target_word_id = word_item.id
            target_morp_list = [x for x in src_item.morp_list if x.word_id == target_word_id]
            sp_pos_list = [x for x in target_morp_list if x.label in split_giho_label]

            if 0 < len(sp_pos_list):
                word_ch_list = []
                char_list = list(word_item.form)
                for ch in char_list:
                    if 0 < len(sp_pos_list):
                        if ch != sp_pos_list[0].form:
                            word_ch_list.append(ch)
                        else:
                            if 0 < len(word_ch_list):
                                concat_ch = "".join(word_ch_list)
                                concat_ch_tokens = tokenizer.tokenize(concat_ch)
                                concat_ch_pos = [p for p in target_morp_list if p.position < sp_pos_list[0].position]
                                for ch_pos in concat_ch_pos:
                                    target_morp_list.remove(ch_pos)
                                concat_ch_pair = (concat_ch, concat_ch_tokens, concat_ch_pos)
                                word_tokens_pos_pair_list.append(concat_ch_pair)
                                word_ch_list.clear()

                            sp_form = sp_pos_list[0].form
                            sp_tokens = tokenizer.tokenize(sp_form)
                            sp_pos = [sp_pos_list[0]]
                            target_morp_list.remove(sp_pos_list[0])
                            sp_pair = (sp_form, sp_tokens, sp_pos)
                            word_tokens_pos_pair_list.append(sp_pair)
                            sp_pos_list.remove(sp_pos_list[0])
                    else:
                        word_ch_list.append(ch)
                if 0 < len(word_ch_list):
                    left_form = "".join(word_ch_list)
                    word_ch_list.clear()
                    left_tokens = tokenizer.tokenize(left_form)
                    left_pos = [p for p in target_morp_list]
                    left_pair = (left_form, left_tokens, left_pos)
                    word_tokens_pos_pair_list.append(left_pair)
            else:
                normal_word_tokens = tokenizer.tokenize(word_item.form)
                normal_pair = (word_item.form, normal_word_tokens, target_morp_list)
                word_tokens_pos_pair_list.append(normal_pair)

        # 명사파생 접미사, 보조사, 주격조사
        # 뒤에서 부터 읽는다.
        '''
            XSN : 명사파생 접미사 (-> 학생'들'의, 선생'님')
            JX : 보조사 (-> 이사장'은')
            JC : 접속 조사 (-> 국무 장관'과')
            JKS : 주격 조사 (-> 위원장'이')
            JKC : 보격 조사 (-> 106주년'이')
            JKG : 관형격 조사 (-> 미군'의')
            JKO : 목적격 조사 (-> 러시아의(관형격) 손'을')
            JKB : 부사격 조사 (-> 7월'에')
        '''
        new_word_tokens_pos_pair_list: List[Tuple[str, List[str], List[str]]] = []
        # VCP -> 긍정지정사
        target_josa = ["XSN", "JX", "JC", "JKS", "JKC", "JKG", "JKO", "JKB", "VCP"]
        # target_josa = ["VCP"]
        target_nn = ["NNG", "NNP", "NNB", "SW"] # 기호 추가, XSN은 VCP만 분리할때
        for wtp_item in word_tokens_pos_pair_list:
            split_idx = -1
            for mp_idx, wtp_mp_item in enumerate(reversed(wtp_item[-1])):
                if wtp_mp_item.label in target_josa:
                    split_idx = len(wtp_item[-1]) - mp_idx - 1
            if -1 != split_idx and 0 != split_idx:
                front_item = wtp_item[-1][:split_idx]
                front_item_nn_list = [x for x in front_item if x.label in target_nn]

                if front_item[-1].label not in target_nn:
                # if 0 >= len(front_item_nn_list):
                    new_word_tokens_pos_pair_list.append(wtp_item)
                    continue
                back_item = wtp_item[-1][split_idx:]

                # wtp_tokens = [t.replace("##", "") for t in wtp_item[1]]
                front_str = "".join([x.form for x in front_item])
                front_tokens = tokenizer.tokenize(front_str)
                back_str = wtp_item[0].replace(front_str, "")
                back_tokens = tokenizer.tokenize(back_str)

                new_front_pair = (front_str, front_tokens, front_item)
                new_back_pair = (back_str, back_tokens, back_item)
                new_word_tokens_pos_pair_list.append(new_front_pair)
                new_word_tokens_pos_pair_list.append(new_back_pair)
            else:
                new_word_tokens_pos_pair_list.append(wtp_item)

        # 전/후 비교를 위해 주석처리
        word_tokens_pos_pair_list = new_word_tokens_pos_pair_list

        # Text Tokens
        text_tokens = []
        for wtp_item in word_tokens_pos_pair_list:
            text_tokens.extend(wtp_item[1])

        # NE
        wtp_pair_len = len(word_tokens_pos_pair_list)
        labels_ids = [ETRI_TAG["O"]] * wtp_pair_len
        LS_ids = [LS_Tag["S"]] * wtp_pair_len
        b_check_use_eojeol = [False for _ in range(wtp_pair_len)]
        for ne_idx, ne_item in enumerate(src_item.ne_list):
            ne_char_list = list(ne_item.text.replace(" ", ""))
            concat_wtp_list = []
            for wtp_idx, wtp_item in enumerate(word_tokens_pos_pair_list):
                if b_check_use_eojeol[wtp_idx]:
                    continue
                for sub_idx in range(wtp_idx + 1, len(word_tokens_pos_pair_list)):
                    concat_wtp = [x[0] for x in word_tokens_pos_pair_list[wtp_idx:sub_idx]]
                    concat_wtp_list.append(("".join(concat_wtp), (wtp_idx, sub_idx)))
            concat_wtp_list = [x for x in concat_wtp_list if "".join(ne_char_list) in x[0]]
            concat_wtp_list.sort(key=lambda x: len(x[0]))
            # print("".join(ne_char_list), concat_wtp_list)
            if 0 >= len(concat_wtp_list):
                continue
            target_index_pair = concat_wtp_list[0][1]

            for bio_idx in range(target_index_pair[0], target_index_pair[1]):
                b_check_use_eojeol[bio_idx] = True
                if bio_idx == target_index_pair[0]:
                    labels_ids[bio_idx] = ETRI_TAG["B-" + ne_item.type]
                else:
                    labels_ids[bio_idx] = ETRI_TAG["I-" + ne_item.type]
                    LS_ids[bio_idx] = LS_Tag["L"]

            for use_idx in range(target_index_pair[1]):
                b_check_use_eojeol[use_idx] = True

        # POS
        pos_tag_ids = [] # [ [POS] * 10 ]
        for wtp_idx, wtp_item in enumerate(word_tokens_pos_pair_list):
            cur_pos_tag = [pos_tag2ids["O"]] * 10

            wtp_morp_list = wtp_item[-1][:10]
            for add_idx, wtp_morp_item in enumerate(wtp_morp_list):
                cur_pos_tag[add_idx] = pos_tag2ids[wtp_morp_item.label]
            pos_tag_ids.append(cur_pos_tag)

        # split_vcp boundary
        eojeol_boundary_list: List[int] = []
        for wtp_ids, wtp_item in enumerate(word_tokens_pos_pair_list):
            token_size = len(wtp_item[1])
            eojeol_boundary_list.append(token_size)

        # Sequence Length
        # 토큰 단위
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        text_tokens.append("[SEP]")
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len-1]
            text_tokens.append("[SEP]")
            valid_token_len = max_len
        else:
            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (max_len - valid_token_len)

        attention_mask = ([1] * valid_token_len) + ([0] * (max_len - valid_token_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        # 어절 단위
        # label_ids
        valid_eojeol_len = 0
        labels_ids.insert(0, ETRI_TAG["O"])
        if eojeol_max_len <= len(labels_ids):
            labels_ids = labels_ids[:eojeol_max_len-1]
            labels_ids.append(ETRI_TAG["O"])
            valid_eojeol_len = eojeol_max_len
        else:
            labels_ids_size = len(labels_ids)
            valid_eojeol_len = labels_ids_size
            for _ in range(eojeol_max_len - labels_ids_size):
                labels_ids.append(ETRI_TAG["O"])

        # LS_ids
        LS_ids.insert(0, LS_Tag["S"])
        if eojeol_max_len <= len(LS_ids):
            LS_ids = LS_ids[:eojeol_max_len-1]
            LS_ids.append(LS_Tag["S"])
        else:
            LS_ids_size = len(LS_ids)
            for _ in range(eojeol_max_len - LS_ids_size):
                LS_ids.append(LS_Tag["S"])

        # pos_tag_ids
        pos_tag_ids.insert(0, [pos_tag2ids["O"]] * 10) # [CLS]
        if eojeol_max_len <= len(pos_tag_ids):
            pos_tag_ids = pos_tag_ids[:eojeol_max_len-1]
            pos_tag_ids.append([pos_tag2ids["O"]] * 10)  # [SEP]
        else:
            pos_tag_ids_size = len(pos_tag_ids)
            for _ in range(eojeol_max_len - pos_tag_ids_size):
                pos_tag_ids.append([pos_tag2ids["O"]] * 10)

        # eojeol_ids
        eojeol_boundary_list.insert(0, 1) # [CLS]
        if eojeol_max_len <= len(eojeol_boundary_list):
            eojeol_boundary_list = eojeol_boundary_list[:eojeol_max_len-1]
            eojeol_boundary_list.append(1) # [SEP]
        else:
            eojeol_boundary_size = len(eojeol_boundary_list)
            eojeol_boundary_list += [0] * (eojeol_max_len - eojeol_boundary_size)

        assert len(input_ids) == max_len, f"{len(input_ids)} + {input_ids}"
        assert len(attention_mask) == max_len, f"{len(attention_mask)} + {attention_mask}"
        assert len(token_type_ids) == max_len, f"{len(token_type_ids)} + {token_type_ids}"
        assert len(labels_ids) == eojeol_max_len, f"{len(labels_ids)} + {labels_ids}"
        assert len(pos_tag_ids) == eojeol_max_len, f"{len(pos_tag_ids)} + {pos_tag_ids}"
        assert len(eojeol_boundary_list) == eojeol_max_len, f"{len(eojeol_boundary_list)} + {eojeol_boundary_list}"
        assert len(LS_ids) == eojeol_max_len, f"{len(LS_ids)} + {LS_ids}"

        # add to npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        #npy_dict["token_seq_len"].append(valid_eojeol_len)
        npy_dict["labels"].append(labels_ids)

        # convert tags
        pos_tag_ids = convert_pos_tag_to_combi_tag(pos_tag_ids)
        npy_dict["pos_tag_ids"].append(pos_tag_ids)

        npy_dict["eojeol_ids"].append(eojeol_boundary_list)
        npy_dict["LS_ids"].append(LS_ids)

        # debug_mode
        if debug_mode:
            # compare - 전체 문장 vs 어절 붙이기
            one_sent_tokenized = tokenizer.tokenize(src_item.text)
            print(src_item.text)
            print(one_sent_tokenized)
            print(text_tokens)
            print("WORD: ")
            print(src_item.word_list)

            print("Unit: WordPiece Token")
            print(f"text_tokens: {text_tokens}")
            # for ii, am, tti in zip(input_ids, attention_mask, token_type_ids):
            #     print(ii, tokenizer.convert_ids_to_tokens([ii]), am, tti)
            print("Unit: Eojeol")
            print(f"seq_len: {valid_eojeol_len} : {len(word_tokens_pos_pair_list)}")
            print(f"label_ids.len: {len(labels_ids)}, pos_tag_ids.len: {len(pos_tag_ids)}")
            print(f"{word_tokens_pos_pair_list}")
            print(f"NE: {src_item.ne_list}")
            print(f"Morp: {src_item.morp_list}")
            print(f"split_vcp boundary: {eojeol_boundary_list}")
            # [(word, [tokens], (begin, end))]
            temp_word_tokens_pos_pair_list = copy.deepcopy(word_tokens_pos_pair_list)
            temp_word_tokens_pos_pair_list.insert(0, ["[CLS]", ["[CLS]"]])
            temp_word_tokens_pos_pair_list.append(["[SEP]", ["[SEP]"]])
            debug_pos_tag_ids = [[pos_ids2tag[x] for x in pos_tag_item] for pos_tag_item in pos_tag_ids]
            for wtpp, la, ls, pti, ej_b in zip(temp_word_tokens_pos_pair_list, labels_ids, LS_ids, debug_pos_tag_ids,
                                               eojeol_boundary_list):
                # pos_array = np.array(pti)
                # if (4 < np.where(pos_array != 'O')[0].size) and (2 <= np.where(pos_array == 'NNP')[0].size):
                print(wtpp[0], ne_ids2tag[la], ls_ids2tag[ls], pti, wtpp[1], ej_b)
            input()

    # save npy_dict
    save_eojeol_npy_dict(npy_dict, len(src_list), save_dir=save_model_dir)

#==============================================================
def convert_pos_tag_to_combi_tag(src_pos: List[List[int]], use_nikl: bool = True):
#==============================================================
    if use_nikl:
        conv_pos_tok2ids = {v: k for k, v in NIKL_POS_TAG.items()}
    else:
        conv_pos_tok2ids = {v: k for k, v in MECAB_POS_TAG.items()}
    ret_pos_list = []

    for eojeol_pos in src_pos:
        new_eojeol_pos =[]
        check_used = [False for _ in range(len(eojeol_pos))]
        for curr_idx, curr_pos_item in enumerate(eojeol_pos):
            # skip process
            if check_used[curr_idx]:
                continue
            if 1 < len(eojeol_pos) and ((conv_pos_tok2ids["NNP"] == eojeol_pos[curr_idx]) or \
                                        (conv_pos_tok2ids["NNG"] == eojeol_pos[curr_idx])):
                if (curr_idx != len(eojeol_pos) - 1):
                    if (conv_pos_tok2ids["NNP"] != eojeol_pos[curr_idx + 1]) and \
                        (conv_pos_tok2ids["NNG"] != eojeol_pos[curr_idx + 1]):
                        new_eojeol_pos.append(curr_pos_item)
                        continue

                for next_idx in range(curr_idx, len(eojeol_pos)):
                    if (conv_pos_tok2ids["NNP"] != eojeol_pos[next_idx]) and \
                            (conv_pos_tok2ids["NNG"] != eojeol_pos[next_idx]):
                        break
                    check_used[next_idx] = True
                new_eojeol_pos.append(conv_pos_tok2ids["CONCAT_NN"])
            else:
                check_used[curr_idx] = True
                new_eojeol_pos.append(curr_pos_item)
            diff_len = len(new_eojeol_pos)
        new_eojeol_pos += [0] * (10 - diff_len)
        ret_pos_list.append(new_eojeol_pos)
        # pos_ids2tag = {k: v for k, v in NIKL_POS_TAG.items()}
        # before_conv = [pos_ids2tag[x] for x in eojeol_pos]
        # after_conv = [pos_ids2tag[x] for x in new_eojeol_pos]
        # if before_conv != after_conv:
        #     print("==============\nBEFORE: ", before_conv, "\n")
        #     print("AFTER: ", after_conv, "\n==============\n")

    '''
    # convert
    for eojeol_pos in src_pos:
        add_idx = 0
        new_eojeol_pos = [0 for _ in range(len(eojeol_pos))]
        check_used = [False for _ in range(len(eojeol_pos))]
        for p_idx, pos_item in enumerate(eojeol_pos):
            is_add = True
            if check_used[p_idx]:
                continue
            elif p_idx == (len(eojeol_pos) - 1):
                new_eojeol_pos[add_idx] = pos_item
                continue

            next_pos_item = eojeol_pos[p_idx+1]
            if conv_pos_tok2ids["NNP"] == pos_item and conv_pos_tok2ids["NNP"] == next_pos_item:
                # [2, 2] -> 49
                new_eojeol_pos[add_idx] = conv_pos_tok2ids["NNP+NNP"]
            elif conv_pos_tok2ids["NNG"] == pos_item and conv_pos_tok2ids["NNG"] == next_pos_item:
                # [1, 1] -> 50
                new_eojeol_pos[add_idx] = conv_pos_tok2ids["NNG+NNG"]
            elif conv_pos_tok2ids["EP"] == pos_item:
                if conv_pos_tok2ids["EC"] == next_pos_item:
                    # [26, 28] -> 51
                    new_eojeol_pos[add_idx] = conv_pos_tok2ids["EP+EC"]
                elif conv_pos_tok2ids["ETM"] == next_pos_item:
                    # [26, 30] -> 52
                    new_eojeol_pos[add_idx] = conv_pos_tok2ids["EP+ETM"]
                else:
                    is_add = False
            elif conv_pos_tok2ids["SN"] == pos_item:
                if conv_pos_tok2ids["NNB"] == next_pos_item:
                    # [44, 3] -> 53
                    new_eojeol_pos[add_idx] = conv_pos_tok2ids["SN+NNB"]
                elif conv_pos_tok2ids["NR"] == next_pos_item:
                    # [44, 5] -> 54
                    new_eojeol_pos[add_idx] = conv_pos_tok2ids["SN+NR"]
                elif conv_pos_tok2ids["SW"] == next_pos_item:
                    # [44, 41] -> 55
                    new_eojeol_pos[add_idx] = conv_pos_tok2ids["SN+SW"]
                else:
                    is_add = False
            else:
                is_add = False

            if is_add:
                check_used[p_idx] = check_used[p_idx+1] = True
            else:
                new_eojeol_pos[add_idx] = pos_item
            add_idx += 1
        # end loop, eojeol_pos
        ret_pos_list.append(new_eojeol_pos)
    #     print(eojeol_pos, " -> ", new_eojeol_pos)
    # input()
    '''

    return ret_pos_list

#==============================================================
def save_eojeol_npy_dict(npy_dict: Dict[str, List], src_list_len, save_dir: str = None):
#==============================================================
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    #npy_dict["token_seq_len"] = np.array(npy_dict["token_seq_len"])
    npy_dict["pos_tag_ids"] = np.array(npy_dict["pos_tag_ids"])
    npy_dict["eojeol_ids"] = np.array(npy_dict["eojeol_ids"])
    npy_dict["LS_ids"] = np.array(npy_dict["LS_ids"])

    # wordpiece 토큰 단위
    print(f"Unit: Tokens")
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    #print(f"seq_len.shape: {npy_dict['token_seq_len'].shape}")

    # 어절 단위
    print(f"Unit: Eojoel")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")
    print(f"eojeol_ids.shape: {npy_dict['eojeol_ids'].shape}")
    print(f"LS_ids.shape: {npy_dict['LS_ids'].shape}")

    # train/dev/test 분할
    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    # train
    train_np = [npy_dict["input_ids"][:train_size],
                npy_dict["attention_mask"][:train_size],
                npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_labels_np = npy_dict["labels"][:train_size]
    #train_seq_len_np = npy_dict["token_seq_len"][:train_size]
    train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]
    train_eojeol_ids_np = npy_dict["eojeol_ids"][:train_size]
    train_LS_ids_np = npy_dict["LS_ids"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_labels_np.shape: {train_labels_np.shape}")
    #print(f"train_seq_len_np.shape: {train_seq_len_np.shape}")
    print(f"train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")
    print(f"train_eojeol_ids_np.shape: {train_eojeol_ids_np.shape}")
    print(f"train_LS_ids_np.shape: {train_LS_ids_np.shape}")

    # dev
    dev_np = [npy_dict["input_ids"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size],
              npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_labels_np = npy_dict["labels"][train_size:valid_size]
    #dev_seq_len_np = npy_dict["token_seq_len"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]
    dev_eojeol_ids_np = npy_dict["eojeol_ids"][train_size:valid_size]
    dev_LS_ids_np = npy_dict["LS_ids"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_labels_np.shape: {dev_labels_np.shape}")
    #print(f"dev_seq_len_np.shape: {dev_seq_len_np.shape}")
    print(f"dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")
    print(f"dev_eojeol_ids_np.shape: {dev_eojeol_ids_np.shape}")
    print(f"dev_LS_ids_np.shape: {dev_LS_ids_np.shape}")

    # test
    test_np = [npy_dict["input_ids"][valid_size:],
               npy_dict["attention_mask"][valid_size:],
               npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_labels_np = npy_dict["labels"][valid_size:]
    #test_seq_len_np = npy_dict["token_seq_len"][valid_size:]
    test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]
    test_eojeol_ids_np = npy_dict["eojeol_ids"][valid_size:]
    test_LS_ids_np = npy_dict["LS_ids"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_labels_np.shape: {test_labels_np.shape}")
    #print(f"test_seq_len_np.shape: {test_seq_len_np.shape}")
    print(f"test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")
    print(f"test_eojeol_ids_np.shape: {test_eojeol_ids_np.shape}")
    print(f"test_LS_ids_np.shape: {test_LS_ids_np.shape}")

    root_path = "../corpus/npy/" + save_dir
    # save input_ids, attention_mask, token_type_ids
    np.save(root_path+"/train", train_np)
    np.save(root_path+"/dev", dev_np)
    np.save(root_path+"/test", test_np)

    # save labels
    np.save(root_path+"/train_labels", train_labels_np)
    np.save(root_path+"/dev_labels", dev_labels_np)
    np.save(root_path+"/test_labels", test_labels_np)

    # save token_seq_len
    # np.save(root_path+"/train_token_seq_len", train_seq_len_np)
    # np.save(root_path+"/dev_token_seq_len", dev_seq_len_np)
    # np.save(root_path+"/test_token_seq_len", test_seq_len_np)

    # save pos_tag_ids
    np.save(root_path+"/train_pos_tag", train_pos_tag_np)
    np.save(root_path+"/dev_pos_tag", dev_pos_tag_np)
    np.save(root_path+"/test_pos_tag", test_pos_tag_np)

    # save eojeol_ids
    np.save(root_path+"/train_eojeol_ids", train_eojeol_ids_np)
    np.save(root_path+"/dev_eojeol_ids", dev_eojeol_ids_np)
    np.save(root_path+"/test_eojeol_ids", test_eojeol_ids_np)

    # save LS_ids
    np.save(root_path+"/train_ls_ids", train_LS_ids_np)
    np.save(root_path+"/dev_ls_ids", dev_eojeol_ids_np)
    np.save(root_path+"/test_ls_ids", test_eojeol_ids_np)

    print(f"[make_gold_corpus_npy][save_eojeol_npy_dict] Complete - Save all npy files !")

#==============================================================
def make_eojeol_and_wordpiece_labels_npy(
        tokenizer_name: str, src_list: List[Sentence],
        max_len: int=128, eojeol_max_len: int=50, debug_mode: bool=False,
        save_model_dir: str = None
):
#==============================================================
    if not save_model_dir:
        print(f"[make_eojeol_datasets_npy] Plz check save_model_dir: {save_model_dir}")
        return

    random.shuffle(src_list)

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        #"token_seq_len": [],
        "pos_tag_ids": [],
        "eojeol_ids": [],
    }

    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in NIKL_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}

    if "bert" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    # Test sentences
    test_str_list = [
        "29·미국·사진", "전창수(42) 두산타워 마케팅팀 차장", "샌드위치→역(逆)샌드위치→신(新)샌드위치….",
        "홍준표·원희룡·나경원 후보가 '3강'을 형성하며 엎치락뒤치락해 왔다.", "P 불투르(Vulture) 인사위원회 위원장은",
        "넙치·굴비·홍어·톳·꼬시래기·굴·홍합", "연준 의장이", "황병서 북한군 총정치국장이 올해 10월 4일",
        "영업익 4482억 ‘깜짝’… LG전자 ‘부활의 노래’", "LG 우규민-삼성 웹스터(대구)",
        "‘김종영 그 절대를 향한’ 전시회", "재산증가액이 3억5000만원이다.", "‘진실·화해를 위한 과거사 정리위원회’",
        "용의자들은 25일 아침 9시께",
    ]

    for proc_idx, src_item in enumerate(src_list):
        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")

        # make (word, token, pos) pair
        # [(word, [tokens], [POS])]
        word_tokens_pos_pair_list: List[Tuple[str, List[str], List[str]]] = []
        separation_giho = ["「", "」", "…", "〔", "〕", "(", ")",
                           "\"", "…", "...", "→", "_", "|", "〈", "〉",
                           "?", "!", "<", ">", "ㆍ", "•", "《", "》",
                           "[", "]", "ㅡ", "+", "“", "”", ";", "·",
                           "‘", "’", "″", "″", "'", "'", "-", "~"]  # , "."]

        for word_idx, word_item in enumerate(src_item.word_list):
            target_word_id = word_item.id
            target_morp_list = [x for x in src_item.morp_list if x.word_id == target_word_id]
            sp_pos_list = [x for x in target_morp_list if x.form in separation_giho]

            if 0 < len(sp_pos_list):
                word_ch_list = []
                char_list = list(word_item.form)
                for ch in char_list:
                    if 0 < len(sp_pos_list):
                        if ch != sp_pos_list[0].form:
                            word_ch_list.append(ch)
                        else:
                            if 0 < len(word_ch_list):
                                concat_ch = "".join(word_ch_list)
                                concat_ch_tokens = tokenizer.tokenize(concat_ch)
                                concat_ch_pos = [p for p in target_morp_list if p.position < sp_pos_list[0].position]
                                for ch_pos in concat_ch_pos:
                                    target_morp_list.remove(ch_pos)
                                concat_ch_pair = (concat_ch, concat_ch_tokens, concat_ch_pos)
                                word_tokens_pos_pair_list.append(concat_ch_pair)
                                word_ch_list.clear()

                            sp_form = sp_pos_list[0].form
                            sp_tokens = tokenizer.tokenize(sp_form)
                            sp_pos = [sp_pos_list[0]]
                            target_morp_list.remove(sp_pos_list[0])
                            sp_pair = (sp_form, sp_tokens, sp_pos)
                            word_tokens_pos_pair_list.append(sp_pair)
                            sp_pos_list.remove(sp_pos_list[0])
                    else:
                        word_ch_list.append(ch)
                if 0 < len(word_ch_list):
                    left_form = "".join(word_ch_list)
                    word_ch_list.clear()
                    left_tokens = tokenizer.tokenize(left_form)
                    left_pos = [p for p in target_morp_list]
                    left_pair = (left_form, left_tokens, left_pos)
                    word_tokens_pos_pair_list.append(left_pair)
            else:
                normal_word_tokens = tokenizer.tokenize(word_item.form)
                normal_pair = (word_item.form, normal_word_tokens, target_morp_list)
                word_tokens_pos_pair_list.append(normal_pair)

        # 명사파생 접미사, 보조사, 주격조사
        # 뒤에서 부터 읽는다.
        '''
            XSN : 명사파생 접미사 (-> 학생'들'의, 선생'님')
            JX : 보조사 (-> 이사장'은')
            JC : 접속 조사 (-> 국무 장관'과')
            JKS : 주격 조사 (-> 위원장'이')
            JKC : 보격 조사 (-> 106주년'이')
            JKG  : 관형격 조사 (-> 미군'의')
            JKO : 목적격 조사 (-> 러시아의(관형격) 손'을')
            JKB : 부사격 조사 (-> 7월'에')
        '''

        new_word_tokens_pos_pair_list: List[Tuple[str, List[str], List[str]]] = []
        # VCP -> 긍정지정사
        target_josa = ["XSN", "JX", "JC", "JKS", "JKC", "JKG", "JKO", "JKB", "VCP"]
        target_nn = ["NNG", "NNP", "NNB"]
        for wtp_item in word_tokens_pos_pair_list:
            split_idx = -1
            for mp_idx, wtp_mp_item in enumerate(reversed(wtp_item[-1])):
                if wtp_mp_item.label in target_josa:
                    split_idx = len(wtp_item[-1]) - mp_idx - 1
            if -1 != split_idx and 0 != split_idx:
                front_item = wtp_item[-1][:split_idx]
                if front_item[-1].label not in target_nn:
                    continue
                back_item = wtp_item[-1][split_idx:]

                # wtp_tokens = [t.replace("##", "") for t in wtp_item[1]]
                front_str = "".join([x.form for x in front_item])
                front_tokens = tokenizer.tokenize(front_str)
                back_str = wtp_item[0].replace(front_str, "")
                back_tokens = tokenizer.tokenize(back_str)

                new_front_pair = (front_str, front_tokens, front_item)
                new_back_pair = (back_str, back_tokens, back_item)
                new_word_tokens_pos_pair_list.append(new_front_pair)
                new_word_tokens_pos_pair_list.append(new_back_pair)
            else:
                new_word_tokens_pos_pair_list.append(wtp_item)
        word_tokens_pos_pair_list = new_word_tokens_pos_pair_list

        # Text Tokens
        text_tokens = []
        for wtp_item in word_tokens_pos_pair_list:
            text_tokens.extend(wtp_item[1])

        # NE
        labels = [ETRI_TAG["O"]] * len(text_tokens)
        start_idx = 0
        for ne_item in src_item.ne_list:
            is_find = False
            for s_idx in range(start_idx, len(text_tokens)):
                if is_find:
                    break
                for word_cnt in range(0, len(text_tokens) - s_idx + 1):
                    concat_text_tokens = text_tokens[s_idx:s_idx + word_cnt]
                    concat_text_tokens = [x.replace("##", "") for x in concat_text_tokens]

                    if "".join(concat_text_tokens) == ne_item.text.replace(" ", ""):
                        # BIO Tagging
                        for bio_idx in range(s_idx, s_idx + word_cnt):
                            if bio_idx == s_idx:
                                labels[bio_idx] = ETRI_TAG["B-" + ne_item.type]
                            else:
                                labels[bio_idx] = ETRI_TAG["I-" + ne_item.type]

                        is_find = True
                        start_idx = s_idx + word_cnt
                        break
        ## end, NE loop

        # POS
        pos_tag_ids = [] # [ [POS] * 10 ]
        for wtp_idx, wtp_item in enumerate(word_tokens_pos_pair_list):
            cur_pos_tag = [pos_tag2ids["O"]] * 10

            wtp_morp_list = wtp_item[-1][:10]
            for add_idx, wtp_morp_item in enumerate(wtp_morp_list):
                cur_pos_tag[add_idx] = pos_tag2ids[wtp_morp_item.label]
            pos_tag_ids.append(cur_pos_tag)

        # Eojeol boundary
        eojeol_boundary_list: List[int] = []
        for wtp_ids, wtp_item in enumerate(word_tokens_pos_pair_list):
            token_size = len(wtp_item[1])
            eojeol_boundary_list.append(token_size)

        # 토큰 단위
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        labels.insert(0, ETRI_TAG["O"])
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len-1]
            labels = labels[:max_len - 1]
            text_tokens.append("[SEP]")
            labels.append(ETRI_TAG["O"])

            valid_token_len = max_len
        else:
            text_tokens.append("[SEP]")
            labels.append(ETRI_TAG["O"])

            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (max_len - valid_token_len)
            labels += [ETRI_TAG["O"]] * (max_len - valid_token_len)

        attention_mask = ([1] * valid_token_len) + ([0] * (max_len - valid_token_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        
        # 어절 단위
        pos_tag_ids.insert(0, [pos_tag2ids["O"]] * 10)  # [CLS]
        if eojeol_max_len <= len(pos_tag_ids):
            pos_tag_ids = pos_tag_ids[:eojeol_max_len - 1]
            pos_tag_ids.append([pos_tag2ids["O"]] * 10)  # [SEP]
        else:
            pos_tag_ids_size = len(pos_tag_ids)
            for _ in range(eojeol_max_len - pos_tag_ids_size):
                pos_tag_ids.append([pos_tag2ids["O"]] * 10)

        # eojeol_ids
        eojeol_boundary_list.insert(0, 1)  # [CLS]
        if eojeol_max_len <= len(eojeol_boundary_list):
            eojeol_boundary_list = eojeol_boundary_list[:eojeol_max_len - 1]
            eojeol_boundary_list.append(1)  # [SEP]
        else:
            eojeol_boundary_size = len(eojeol_boundary_list)
            eojeol_boundary_list += [0] * (eojeol_max_len - eojeol_boundary_size)

        # Check size
        assert len(input_ids) == max_len, f"{len(input_ids)} + {input_ids}"
        assert len(attention_mask) == max_len, f"{len(attention_mask)} + {attention_mask}"
        assert len(token_type_ids) == max_len, f"{len(token_type_ids)} + {token_type_ids}"
        assert len(labels) == max_len, f"{len(labels)} + {labels}"
        assert len(pos_tag_ids) == eojeol_max_len, f"{len(pos_tag_ids)} + {pos_tag_ids}"
        assert len(eojeol_boundary_list) == eojeol_max_len, f"{len(eojeol_boundary_list)} + {eojeol_boundary_list}"

        # add to npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        #npy_dict["token_seq_len"].append(valid_eojeol_len)  # split_vcp !
        npy_dict["labels"].append(labels)

        # convert pos tags
        pos_tag_ids = convert_pos_tag_to_combi_tag(pos_tag_ids)
        npy_dict["pos_tag_ids"].append(pos_tag_ids)

        npy_dict["eojeol_ids"].append(eojeol_boundary_list)

        # debug mode
        if debug_mode:
            print(f"Sent: {src_item.text}")
            print(f"NE: {src_item.ne_list}")
            print(f"Morp: {src_item.morp_list}")
            temp_word_token_pos_pair_list = copy.deepcopy(word_tokens_pos_pair_list)
            temp_word_token_pos_pair_list.insert(0, ["[CLS]", ["[CLS]"]])
            temp_word_token_pos_pair_list.append(["[SEP]", ["[SEP]"]])
            debug_pos_tag_ids = [[pos_ids2tag[x] for x in pos_tag_item] for pos_tag_item in pos_tag_ids]
            # Eojeol
            for wtpp, pti, ej_b in  zip(temp_word_token_pos_pair_list, debug_pos_tag_ids, eojeol_boundary_list):
                print(wtpp[0], pti, wtpp[1], ej_b)
            # Token
            for ii, la in zip(input_ids, labels):
                print(tokenizer.convert_ids_to_tokens([ii]), ne_ids2tag[la])
            input()

    # save npy_dict
    save_eojeol_and_wordpiece_labels_npy(npy_dict, len(src_list), save_dir=save_model_dir)

#==============================================================
def save_eojeol_and_wordpiece_labels_npy(npy_dict: Dict[str, List], src_list_len, save_dir: str = None):
#==============================================================
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    #npy_dict["token_seq_len"] = np.array(npy_dict["token_seq_len"])
    npy_dict["pos_tag_ids"] = np.array(npy_dict["pos_tag_ids"])
    npy_dict["eojeol_ids"] = np.array(npy_dict["eojeol_ids"])

    # wordpiece 토큰 단위
    print(f"Unit: Tokens")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    #print(f"token_seq_len.shape: {npy_dict['token_seq_len'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")

    # 어절 단위
    print(f"Unit: Eojoel")
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")
    print(f"eojeol_ids.shape: {npy_dict['eojeol_ids'].shape}")

    # train/dev/test 분할
    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    # train
    train_np = [npy_dict["input_ids"][:train_size],
                npy_dict["attention_mask"][:train_size],
                npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_labels_np = npy_dict["labels"][:train_size]
    #train_seq_len_np = npy_dict["token_seq_len"][:train_size]
    train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]
    train_eojeol_ids_np = npy_dict["eojeol_ids"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_labels_np.shape: {train_labels_np.shape}")
    #print(f"train_seq_len_np.shape: {train_seq_len_np.shape}")
    print(f"train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")
    print(f"train_eojeol_ids_np.shape: {train_eojeol_ids_np.shape}")

    # dev
    dev_np = [npy_dict["input_ids"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size],
              npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_labels_np = npy_dict["labels"][train_size:valid_size]
    #dev_seq_len_np = npy_dict["token_seq_len"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]
    dev_eojeol_ids_np = npy_dict["eojeol_ids"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_labels_np.shape: {dev_labels_np.shape}")
    #print(f"dev_seq_len_np.shape: {dev_seq_len_np.shape}")
    print(f"dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")
    print(f"dev_eojeol_ids_np.shape: {dev_eojeol_ids_np.shape}")

    # test
    test_np = [npy_dict["input_ids"][valid_size:],
               npy_dict["attention_mask"][valid_size:],
               npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_labels_np = npy_dict["labels"][valid_size:]
    #test_seq_len_np = npy_dict["token_seq_len"][valid_size:]
    test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]
    test_eojeol_ids_np = npy_dict["eojeol_ids"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_labels_np.shape: {test_labels_np.shape}")
    #print(f"test_seq_len_np.shape: {test_seq_len_np.shape}")
    print(f"test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")
    print(f"test_eojeol_ids_np.shape: {test_eojeol_ids_np.shape}")

    root_path = "../corpus/npy/" + save_dir
    # save input_ids, attention_mask, token_type_ids
    np.save(root_path+"/train", train_np)
    np.save(root_path+"/dev", dev_np)
    np.save(root_path+"/test", test_np)

    # save labels
    np.save(root_path+"/train_labels", train_labels_np)
    np.save(root_path+"/dev_labels", dev_labels_np)
    np.save(root_path+"/test_labels", test_labels_np)

    # save seq_len
    # np.save(root_path+"/train_token_seq_len", train_seq_len_np)
    # np.save(root_path+"/dev_token_seq_len", dev_seq_len_np)
    # np.save(root_path+"/test_token_seq_len", test_seq_len_np)

    # save pos_tag_ids
    np.save(root_path+"/train_pos_tag", train_pos_tag_np)
    np.save(root_path+"/dev_pos_tag", dev_pos_tag_np)
    np.save(root_path+"/test_pos_tag", test_pos_tag_np)

    # save eojeol_ids
    np.save(root_path+"/train_eojeol_ids", train_eojeol_ids_np)
    np.save(root_path+"/dev_eojeol_ids", dev_eojeol_ids_np)
    np.save(root_path+"/test_eojeol_ids", test_eojeol_ids_np)

    print(f"[make_gold_corpus_npy][save_eojeol_npy_dict] Complete - Save all npy files !")

#==============================================================
def make_not_split_jx_eojeol_datasets_npy(
    tokenizer_name: str, src_list: List[Sentence],
    max_len: int = 128, eojeol_max_len: int = 50, debug_mode: bool = False,
    save_model_dir: str = None, split_vcp: bool = False
):
#==============================================================
    if not save_model_dir:
        print(f"[make_eojeol_datasets_npy] Plz check save_model_dir: {save_model_dir}")
        return

    random.shuffle(src_list)
    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        #"token_seq_len": [],
        "pos_tag_ids": [],
        "eojeol_ids": [],
        "LS_ids": [],
        # "entity_ids": [], # for token_type_ids (bert segment embedding)
    }
    pos_tag2ids = {v: int(k) for k, v in NIKL_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in NIKL_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}
    ls_ids2tag = {v: k for k, v in LS_Tag.items()}

    if "bert" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    # Test sentences
    test_str_list = [
        "29·미국·사진", "전창수(42) 두산타워 마케팅팀 차장", "샌드위치→역(逆)샌드위치→신(新)샌드위치….",
        "홍준표·원희룡·나경원 후보가 '3강'을 형성하며 엎치락뒤치락해 왔다.", "P 불투르(Vulture) 인사위원회 위원장은",
        "넙치·굴비·홍어·톳·꼬시래기·굴·홍합", "연준 의장이", "황병서 북한군 총정치국장이 올해 10월 4일",
        "영업익 4482억 ‘깜짝’… LG전자 ‘부활의 노래’", "LG 우규민-삼성 웹스터(대구)",
        "‘김종영 그 절대를 향한’ 전시회", "재산증가액이 3억5000만원이다.", "‘진실·화해를 위한 과거사 정리위원회’",
        "용의자들은 25일 아침 9시께", "해외50여 개국에서 5500회 이상 공연하며 사물놀이",
        "REDD는 열대우림 등 산림자원을 보호하는 개도국이나",
        "2010년 12월부터 이미 가중 처벌을 시행하는 어린이 보호구역의 교통사고 발생 건수는",
        "금리설계형의 경우 변동금리(6개월 변동 코픽스 연동형)는", "지난해 3월 현재 카타르 수도 도하의 여성 교도소 수감자의"
        "'중국 편'이라고 믿었던 박 대통령에게"
    ]
    # test_str_list = [
    #     "'중국 편'이라고 믿었던 박 대통령에게",
    # ]

    for proc_idx, src_item in enumerate(src_list):
        # Test
        if debug_mode:
            is_test_str = False
            for test_str in test_str_list:
                if test_str in src_item.text:
                    is_test_str = True
            if not is_test_str:
                continue

        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")

        # make (word, token, pos) pair
        # [(word, [tokens], [POS])]
        word_tokens_pos_pair_list: List[Tuple[str, List[str], List[str]]] = []
        # separation_giho = ["「", "」", "…", "〔", "〕", "(", ")",
        #                    "\"", "…", "...", "→", "_", "|", "〈", "〉",
        #                    "?", "!", "<", ">", "ㆍ", "•", "《", "》",
        #                    "[", "]", "ㅡ", "+", "“", "”", ";", "·",
        #                    "‘", "’", "″", "″", "'", "'", "-", "~", ",", "."]
        split_giho_label = [
            "SF", "SP", "SS", "SE", "SO", "SW"
        ]

        for word_idx, word_item in enumerate(src_item.word_list):
            target_word_id = word_item.id
            target_morp_list = [x for x in src_item.morp_list if x.word_id == target_word_id]
            sp_pos_list = [x for x in target_morp_list if x.label in split_giho_label]

            if 0 < len(sp_pos_list):
                word_ch_list = []
                char_list = list(word_item.form)
                for ch in char_list:
                    if 0 < len(sp_pos_list):
                        if ch != sp_pos_list[0].form:
                            word_ch_list.append(ch)
                        else:
                            if 0 < len(word_ch_list):
                                concat_ch = "".join(word_ch_list)
                                concat_ch_tokens = tokenizer.tokenize(concat_ch)
                                concat_ch_pos = [p for p in target_morp_list if p.position < sp_pos_list[0].position]
                                for ch_pos in concat_ch_pos:
                                    target_morp_list.remove(ch_pos)
                                concat_ch_pair = (concat_ch, concat_ch_tokens, concat_ch_pos)
                                word_tokens_pos_pair_list.append(concat_ch_pair)
                                word_ch_list.clear()

                            sp_form = sp_pos_list[0].form
                            sp_tokens = tokenizer.tokenize(sp_form)
                            sp_pos = [sp_pos_list[0]]
                            target_morp_list.remove(sp_pos_list[0])
                            sp_pair = (sp_form, sp_tokens, sp_pos)
                            word_tokens_pos_pair_list.append(sp_pair)
                            sp_pos_list.remove(sp_pos_list[0])
                    else:
                        word_ch_list.append(ch)
                if 0 < len(word_ch_list):
                    left_form = "".join(word_ch_list)
                    word_ch_list.clear()
                    left_tokens = tokenizer.tokenize(left_form)
                    left_pos = [p for p in target_morp_list]
                    left_pair = (left_form, left_tokens, left_pos)
                    word_tokens_pos_pair_list.append(left_pair)
            else:
                normal_word_tokens = tokenizer.tokenize(word_item.form)
                normal_pair = (word_item.form, normal_word_tokens, target_morp_list)
                word_tokens_pos_pair_list.append(normal_pair)

        # 명사파생 접미사, 보조사, 주격조사
        # 뒤에서 부터 읽는다.
        '''
            XSN : 명사파생 접미사 (-> 학생'들'의, 선생'님')
            JX : 보조사 (-> 이사장'은')
            JC : 접속 조사 (-> 국무 장관'과')
            JKS : 주격 조사 (-> 위원장'이')
            JKC : 보격 조사 (-> 106주년'이')
            JKG : 관형격 조사 (-> 미군'의')
            JKO : 목적격 조사 (-> 러시아의(관형격) 손'을')
            JKB : 부사격 조사 (-> 7월'에')
        '''
        if split_vcp:
            new_word_tokens_pos_pair_list: List[Tuple[str, List[str], List[str]]] = []
            # VCP -> 긍정지정사
            target_josa = ["XSN", "JX", "JC", "JKS", "JKC", "JKG", "JKO", "JKB", "VCP"]
            # target_josa = ["VCP"]
            target_nn = ["NNG", "NNP", "NNB", "SW"]  # 기호 추가, XSN은 VCP만 분리할때
            for wtp_item in word_tokens_pos_pair_list:
                split_idx = -1
                for mp_idx, wtp_mp_item in enumerate(reversed(wtp_item[-1])):
                    if wtp_mp_item.label in target_josa:
                        split_idx = len(wtp_item[-1]) - mp_idx - 1
                if -1 != split_idx and 0 != split_idx:
                    front_item = wtp_item[-1][:split_idx]
                    front_item_nn_list = [x for x in front_item if x.label in target_nn]

                    # if front_item[-1].label not in target_nn:
                    if 0 >= len(front_item_nn_list):
                        new_word_tokens_pos_pair_list.append(wtp_item)
                        continue
                    back_item = wtp_item[-1][split_idx:]

                    # wtp_tokens = [t.replace("##", "") for t in wtp_item[1]]
                    front_str = "".join([x.form for x in front_item])
                    front_tokens = tokenizer.tokenize(front_str)
                    back_str = wtp_item[0].replace(front_str, "")
                    back_tokens = tokenizer.tokenize(back_str)

                    new_front_pair = (front_str, front_tokens, front_item)
                    new_back_pair = (back_str, back_tokens, back_item)
                    new_word_tokens_pos_pair_list.append(new_front_pair)
                    new_word_tokens_pos_pair_list.append(new_back_pair)
                else:
                    new_word_tokens_pos_pair_list.append(wtp_item)
            # 전/후 비교를 위해 주석처리
            word_tokens_pos_pair_list = new_word_tokens_pos_pair_list

        # Text Tokens
        text_tokens = []
        for wtp_item in word_tokens_pos_pair_list:
            text_tokens.extend(wtp_item[1])

        # NE
        wtp_pair_len = len(word_tokens_pos_pair_list)
        labels_ids = [ETRI_TAG["O"]] * wtp_pair_len
        LS_ids = [LS_Tag["S"]] * wtp_pair_len
        b_check_use_eojeol = [False for _ in range(wtp_pair_len)]
        for ne_idx, ne_item in enumerate(src_item.ne_list):
            ne_char_list = list(ne_item.text.replace(" ", ""))
            concat_wtp_list = []
            for wtp_idx, wtp_item in enumerate(word_tokens_pos_pair_list):
                if b_check_use_eojeol[wtp_idx]:
                    continue
                for sub_idx in range(wtp_idx+1, len(word_tokens_pos_pair_list)):
                    concat_wtp = [x[0] for x in word_tokens_pos_pair_list[wtp_idx:sub_idx]]
                    concat_wtp_list.append(("".join(concat_wtp), (wtp_idx, sub_idx)))
            concat_wtp_list = [x for x in concat_wtp_list if "".join(ne_char_list) in x[0]]
            concat_wtp_list.sort(key=lambda x: len(x[0]))
            # print("".join(ne_char_list), concat_wtp_list)
            if 0 >= len(concat_wtp_list):
                continue
            target_index_pair = concat_wtp_list[0][1]

            for bio_idx in range(target_index_pair[0], target_index_pair[1]):
                if bio_idx == target_index_pair[0]:
                    labels_ids[bio_idx] = ETRI_TAG["B-" + ne_item.type]
                else:
                    labels_ids[bio_idx] = ETRI_TAG["I-" + ne_item.type]
                    LS_ids[bio_idx] = LS_Tag["L"]

            for use_idx in range(target_index_pair[1]):
                b_check_use_eojeol[use_idx] = True

        # POS
        pos_tag_ids = [] # [ [POS] * 10 ]
        for wtp_idx, wtp_item in enumerate(word_tokens_pos_pair_list):
            cur_pos_tag = [pos_tag2ids["O"]] * 10

            wtp_morp_list = wtp_item[-1][:10]
            for add_idx, wtp_morp_item in enumerate(wtp_morp_list):
                cur_pos_tag[add_idx] = pos_tag2ids[wtp_morp_item.label]
            pos_tag_ids.append(cur_pos_tag)

        # split_vcp boundary
        eojeol_boundary_list: List[int] = []
        for wtp_ids, wtp_item in enumerate(word_tokens_pos_pair_list):
            token_size = len(wtp_item[1])
            eojeol_boundary_list.append(token_size)

        # Sequence Length
        # 토큰 단위
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        text_tokens.append("[SEP]")
        if max_len <= len(text_tokens):
            text_tokens = text_tokens[:max_len-1]
            text_tokens.append("[SEP]")
            valid_token_len = max_len
        else:
            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (max_len - valid_token_len)

        attention_mask = ([1] * valid_token_len) + ([0] * (max_len - valid_token_len))
        token_type_ids = [0] * max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        # 어절 단위
        # label_ids
        valid_eojeol_len = 0
        labels_ids.insert(0, ETRI_TAG["O"])
        if eojeol_max_len <= len(labels_ids):
            labels_ids = labels_ids[:eojeol_max_len-1]
            labels_ids.append(ETRI_TAG["O"])
            valid_eojeol_len = eojeol_max_len
        else:
            labels_ids_size = len(labels_ids)
            valid_eojeol_len = labels_ids_size
            for _ in range(eojeol_max_len - labels_ids_size):
                labels_ids.append(ETRI_TAG["O"])

        # LS_ids
        LS_ids.insert(0, LS_Tag["S"])
        if eojeol_max_len <= len(LS_ids):
            LS_ids = LS_ids[:eojeol_max_len-1]
            LS_ids.append(LS_Tag["S"])
        else:
            LS_ids_size = len(LS_ids)
            for _ in range(eojeol_max_len - LS_ids_size):
                LS_ids.append(LS_Tag["S"])

        # pos_tag_ids
        pos_tag_ids.insert(0, [pos_tag2ids["O"]] * 10) # [CLS]
        if eojeol_max_len <= len(pos_tag_ids):
            pos_tag_ids = pos_tag_ids[:eojeol_max_len-1]
            pos_tag_ids.append([pos_tag2ids["O"]] * 10)  # [SEP]
        else:
            pos_tag_ids_size = len(pos_tag_ids)
            for _ in range(eojeol_max_len - pos_tag_ids_size):
                pos_tag_ids.append([pos_tag2ids["O"]] * 10)

        # eojeol_ids
        eojeol_boundary_list.insert(0, 1) # [CLS]
        if eojeol_max_len <= len(eojeol_boundary_list):
            eojeol_boundary_list = eojeol_boundary_list[:eojeol_max_len-1]
            eojeol_boundary_list.append(1) # [SEP]
        else:
            eojeol_boundary_size = len(eojeol_boundary_list)
            eojeol_boundary_list += [0] * (eojeol_max_len - eojeol_boundary_size)

        assert len(input_ids) == max_len, f"{len(input_ids)} + {input_ids}"
        assert len(attention_mask) == max_len, f"{len(attention_mask)} + {attention_mask}"
        assert len(token_type_ids) == max_len, f"{len(token_type_ids)} + {token_type_ids}"
        assert len(labels_ids) == eojeol_max_len, f"{len(labels_ids)} + {labels_ids}"
        assert len(pos_tag_ids) == eojeol_max_len, f"{len(pos_tag_ids)} + {pos_tag_ids}"
        assert len(eojeol_boundary_list) == eojeol_max_len, f"{len(eojeol_boundary_list)} + {eojeol_boundary_list}"
        assert len(LS_ids) == eojeol_max_len, f"{len(LS_ids)} + {LS_ids}"

        # add to npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        #npy_dict["token_seq_len"].append(valid_eojeol_len)
        npy_dict["labels"].append(labels_ids)

        # convert tags
        pos_tag_ids = convert_pos_tag_to_combi_tag(pos_tag_ids)
        npy_dict["pos_tag_ids"].append(pos_tag_ids)

        npy_dict["eojeol_ids"].append(eojeol_boundary_list)
        npy_dict["LS_ids"].append(LS_ids)

        # debug_mode
        if debug_mode:
            # compare - 전체 문장 vs 어절 붙이기
            one_sent_tokenized = tokenizer.tokenize(src_item.text)
            print(src_item.text)
            print(one_sent_tokenized)
            print(text_tokens)
            print("WORD: ")
            print(src_item.word_list)

            print("Unit: WordPiece Token")
            print(f"text_tokens: {text_tokens}")
            # for ii, am, tti in zip(input_ids, attention_mask, token_type_ids):
            #     print(ii, tokenizer.convert_ids_to_tokens([ii]), am, tti)
            print("Unit: Eojeol")
            print(f"seq_len: {valid_eojeol_len} : {len(word_tokens_pos_pair_list)}")
            print(f"label_ids.len: {len(labels_ids)}, pos_tag_ids.len: {len(pos_tag_ids)}")
            print(f"{word_tokens_pos_pair_list}")
            print(f"NE: {src_item.ne_list}")
            print(f"Morp: {src_item.morp_list}")
            print(f"split_vcp boundary: {eojeol_boundary_list}")
            # [(word, [tokens], (begin, end))]
            temp_word_tokens_pos_pair_list = copy.deepcopy(word_tokens_pos_pair_list)
            temp_word_tokens_pos_pair_list.insert(0, ["[CLS]", ["[CLS]"]])
            temp_word_tokens_pos_pair_list.append(["[SEP]", ["[SEP]"]])
            debug_pos_tag_ids = [[pos_ids2tag[x] for x in pos_tag_item] for pos_tag_item in pos_tag_ids]
            for wtpp, la, ls, pti, ej_b in zip(temp_word_tokens_pos_pair_list, labels_ids, LS_ids, debug_pos_tag_ids,
                                               eojeol_boundary_list):
                # pos_array = np.array(pti)
                # if (4 < np.where(pos_array != 'O')[0].size) and (2 <= np.where(pos_array == 'NNP')[0].size):
                print(wtpp[0], ne_ids2tag[la], ls_ids2tag[ls], pti, wtpp[1], ej_b)
            input()

    # save npy_dict
    save_eojeol_npy_dict(npy_dict, len(src_list), save_dir=save_model_dir)

### MAIN ###
if "__main__" == __name__:
    print("[gold_corpus_npy_maker] __main__ !")

    all_sent_list = []
    # with open("../corpus/pkl/NIKL_ne_pos.pkl", mode="rb") as pkl_file:
    with open("../corpus/pkl/NIKL_ne_pos.pkl", mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[make_gold_corpus_npy][__main__] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)
    all_sent_list = conv_NIKL_pos_giho_category(all_sent_list, is_status_nn=False, is_verb_nn=False)

    # make npy
    is_use_external_dict = False
    hash_dict = None
    if is_use_external_dict:
        hash_dict = make_dict_hash_table(dict_path="../우리말샘_dict.pkl")

    '''
        electra : monologg/koelectra-base-v3-discriminator
        bert : klue/bert-base
        roberta : klue/roberta-base
    '''
    # make_wordpiece_npy(tokenizer_name="monologg/koelectra-base-v3-discriminator",
    #                    src_list=all_sent_list, max_len=128, debug_mode=False, save_model_dir="electra",
    #                    max_pos_nums=10)

    # make_eojeol_datasets_npy(tokenizer_name="monologg/koelectra-base-v3-discriminator",
    #                          src_list=all_sent_list, max_len=128, debug_mode=True,
    #                          save_model_dir="eojeol_electra")

    make_not_split_jx_eojeol_datasets_npy(tokenizer_name="monologg/koelectra-base-v3-discriminator",
                                          src_list=all_sent_list, max_len=128, debug_mode=False,
                                          save_model_dir="eojeol_vcp_electra", split_vcp=True)

    # make_eojeol_and_wordpiece_labels_npy(tokenizer_name="monologg/koelectra-base-v3-discriminator",
    #                                      src_list=all_sent_list, max_len=128, debug_mode=False,
    #                                      save_model_dir="eojeol2wp_electra")