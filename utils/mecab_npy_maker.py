import copy
import random
import numpy as np
import pickle

from dataclasses import dataclass, field
from collections import deque

import platform
if "Windows" == platform.system():
    from eunjeon import Mecab # Windows
else:
    from konlpy.tag import Mecab # Linux

from tag_def import ETRI_TAG, NIKL_POS_TAG, MECAB_POS_TAG
from data_def import Sentence, NE, Morp, Word
from mecab_utils import (
    convert_morp_connected_tokens, convert_character_pos_tokens,
    convert_wordpiece_pos_tokens
)

from gold_corpus_npy_maker import (
    convert_pos_tag_to_combi_tag,
    conv_NIKL_pos_giho_category,
    conv_TTA_ne_category
)

from typing import List, Dict, Tuple
from transformers import AutoTokenizer, ElectraTokenizer
from tokenization_kocharelectra import KoCharElectraTokenizer

# Character Level
import unicodedata
from jamo import h2j, j2hcj
from string import ascii_lowercase
import re

# Test Sentences
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
    "'중국 편'이라고 믿었던 박 대통령에게", "2001년 한·미 주둔군지위협정(소파·SOFA)"
]
test_str_list = ["김행 청와대 대변인은 이날"]

## Structured
@dataclass
class Tok_Pos:
    tokens: List[str] = field(default=list)
    pos: List[str] = field(default=list)
    ne: str = "O"

@dataclass
class Mecab_Item:
    word: str = ""
    ne: str = "O"
    tok_pos_list: List[Tok_Pos] = field(default=list)

## Global
random.seed(42)
np.random.seed(42)

#==========================================================================================
def load_ne_entity_list(src_path: str = ""):
#==========================================================================================
    all_sent_list = []

    with open(src_path, mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[mecab_npy_maker][load_ne_entity_list] all_sent_list size: {len(all_sent_list)}")
    all_sent_list = conv_TTA_ne_category(all_sent_list)

    return all_sent_list

#==========================================================================================
def convert_mecab_pos(sentence: str, src_word_list: List[Word]):
#==========================================================================================
    mecab = Mecab()

    # ret_word_pair : (word, (word, pos))
    ret_word_pair_list = []

    mecab_res = mecab.pos(sentence)
    for mecab_pair in mecab_res:
        ret_word_pair_list.append((mecab_pair[0], [(mecab_pair[0], mecab_pair[1].split("+"))]))

    # for word_idx, word_item in enumerate(src_word_list):
    #     res_list = mecab.pos(word_item.form)
    #     new_res_list = []
    #
    #     for res in res_list:
    #         new_res_list.append((res[0], res[1].split("+")))
    #     ret_word_pair_list.append((word_item.form, new_res_list))

    return ret_word_pair_list

#==========================================================================================
def tokenize_mecab_pair_unit_pos(mecab_pair_list, tokenizer):
#==========================================================================================
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

    return tokenized_mecab_list

#==========================================================================================
def make_mecab_eojeol_npy(
        tokenizer_name: str, src_list: List[Sentence],
        token_max_len: int = 128, eojeol_max_len: int = 50,
        debug_mode: bool = False, josa_split: bool = False,
        save_model_dir: str = None
):
#==========================================================================================
    if not save_model_dir:
        print(f"[mecab_npy_maker] Plz check save_model_dir: {save_model_dir}")
        return

    # shuffle
    random.shuffle(src_list)

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "pos_tag_ids": [],
        "eojeol_ids": [],
    }

    pos_tag2ids = {v: int(k) for k, v in MECAB_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in MECAB_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}

    # Tokenizer
    if "bert" in tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    # Test Sentences
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

        # 매캡을 쓰면 모두의 말뭉치 morp의 word_id 정보는 사용할 수 없음
        extract_ne_list = src_item.ne_list
        # [ word, [(word, pos)] ]
        mecab_word_pair = convert_mecab_pos(sentence=src_item.text, src_word_list=src_item.word_list)
        mecab_item_list = tokenize_mecab_pair_unit_pos(mecab_word_pair, tokenizer)

        split_giho = ["SF", "SE", "SSO", "SSC", "SC", "SY"]
        new_mecab_item_list: List[Mecab_Item] = []
        for mecab_item in mecab_item_list:
            temp_save_tok_pos = []
            for tok_pos in mecab_item.tok_pos_list:
                if tok_pos.pos[0] in split_giho:
                    if 0 < len(temp_save_tok_pos):
                        concat_word = "".join(["".join(x.tokens).replace("##", "") for x in temp_save_tok_pos])
                        new_mecab_item = Mecab_Item(word=concat_word,
                                                    tok_pos_list=temp_save_tok_pos)
                        new_mecab_item_list.append(copy.deepcopy(new_mecab_item))
                        temp_save_tok_pos.clear()
                        if 0 < len(tok_pos.tokens):
                            new_mecab_item_list.append(copy.deepcopy(Mecab_Item(word=tok_pos.tokens[0],
                                                                                tok_pos_list=[tok_pos])))
                else:
                    temp_save_tok_pos.append(tok_pos)
            if 0 < len(temp_save_tok_pos):
                concat_word = "".join(["".join(x.tokens).replace("##", "") for x in temp_save_tok_pos])
                new_mecab_item_list.append(copy.deepcopy(Mecab_Item(word=concat_word,
                                                                    tok_pos_list=temp_save_tok_pos)))
        # if josa_split:
        if josa_split:
            split_josa_mecab_item_list: List[Mecab_Item] = []
            split_target_morp = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"]
            target_nn = ["NNG", "NNP", "NNB"]
            for mecab_item in new_mecab_item_list:
                split_idx = - 1
                for tp_idx, tp_item in enumerate(reversed(mecab_item.tok_pos_list)):
                    filter_morp_list = [tp_i for tp_i in tp_item.pos if tp_i in split_target_morp]
                    if 0 < len(filter_morp_list):
                        split_idx = len(mecab_item.tok_pos_list) - tp_idx - 1

                if -1 != split_idx and 0 != split_idx:
                    front_item = mecab_item.tok_pos_list[:split_idx]
                    front_item_pos_list = []
                    for f_item in front_item:
                        for f_p in f_item.pos:
                            front_item_pos_list.append(f_p)

                    front_item_nn_list = [tp_i for tp_i in front_item_pos_list if tp_i in target_nn]
                    if 0 >= len(front_item_nn_list) or front_item_nn_list[-1] not in target_nn:
                        split_josa_mecab_item_list.append(mecab_item)
                        continue
                    back_item = mecab_item.tok_pos_list[split_idx:]

                    front_tokens = []
                    for f_t in front_item:
                        front_tokens.extend(f_t.tokens)
                    front_str = "".join(front_tokens).replace("##", "")
                    split_josa_mecab_item_list.append(copy.deepcopy(Mecab_Item(word=front_str, tok_pos_list=front_item)))

                    back_tokens = []
                    for b_t in back_item:
                        back_tokens.extend(b_t.tokens)
                    back_str = "".join(back_tokens).replace("##", "")
                    split_josa_mecab_item_list.append(copy.deepcopy(Mecab_Item(word=back_str, tok_pos_list=back_item)))
                else:
                    split_josa_mecab_item_list.append(mecab_item)
            new_mecab_item_list = split_josa_mecab_item_list

        tok_pos_item_list = []
        for mecab_item in new_mecab_item_list:
            tok_pos_item_list.extend(mecab_item.tok_pos_list)

        # NE - 어절 단위
        b_check_use = [False for _ in range(len(tok_pos_item_list))]
        for ne_idx, ne_item in enumerate(extract_ne_list):
            ne_char_list = list(ne_item.text.replace(" ", ""))
            concat_item_list = []
            for mec_idx, mecab_item in enumerate(new_mecab_item_list):
                if b_check_use[mec_idx]:
                    continue
                for sub_idx in range(mec_idx + 1, len(new_mecab_item_list)):
                    concat_word = ["".join(x.word).replace("##", "") for x in new_mecab_item_list[mec_idx:sub_idx]]
                    concat_item_list.append(("".join(concat_word), (mec_idx, sub_idx)))
            concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
            concat_item_list.sort(key=lambda x: len(x[0]))
            if 0 >= len(concat_item_list):
                continue
            target_idx_pair = concat_item_list[0][1]
            for bio_idx in range(target_idx_pair[0], target_idx_pair[1]):
                b_check_use[bio_idx] = True
                if bio_idx == target_idx_pair[0]:
                    new_mecab_item_list[bio_idx].ne = "B-" + ne_item.type
                else:
                    new_mecab_item_list[bio_idx].ne = "I-" + ne_item.type

        # # NE - 토큰 단위
        # b_check_use = [False for _ in range(len(tok_pos_item_list))]
        # for ne_idx, ne_item in enumerate(extract_ne_list):
        #     ne_char_list = list(ne_item.text.replace(" ", ""))
        #     concat_item_list = []
        #     for mec_idx, mecab_item in enumerate(tok_pos_item_list):
        #         if b_check_use[mec_idx]:
        #             continue
        #         for sub_idx in range(mec_idx + 1, len(tok_pos_item_list)):
        #             concat_word = ["".join(x.tokens).replace("##", "") for x in tok_pos_item_list[mec_idx:sub_idx]]
        #             concat_item_list.append(("".join(concat_word), (mec_idx, sub_idx)))
        #     concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
        #     concat_item_list.sort(key=lambda x: len(x[0]))
        #     if 0 >= len(concat_item_list):
        #         continue
        #     target_idx_pair = concat_item_list[0][1]
        #
        #     for bio_idx in range(target_idx_pair[0], target_idx_pair[1]):
        #         b_check_use[bio_idx] = True
        #         if bio_idx == target_idx_pair[0]:
        #             tok_pos_item_list[bio_idx].ne = "B-" + ne_item.type
        #         else:
        #             tok_pos_item_list[bio_idx].ne = "I-" + ne_item.type

        text_tokens = []
        label_ids = []
        pos_ids = []
        for tp_idx, tok_pos in enumerate(tok_pos_item_list):
            for ft_idx, flat_tok in enumerate(tok_pos.tokens):
                text_tokens.append(flat_tok)

        for mec_idx, new_mecab_item in enumerate(new_mecab_item_list):
            label_ids.append(ETRI_TAG[new_mecab_item.ne])
            conv_pos = []
            for mecab_item in new_mecab_item.tok_pos_list:
                filter_pos = [x if "UNKNOWN" != x and "NA" != x and "UNA" != x and "VSV" != x else "O" for x in mecab_item.pos]
                conv_pos.extend([pos_tag2ids[x] for x in filter_pos])
            if 10 > len(conv_pos):
                diff_len = (10 - len(conv_pos))
                conv_pos += [pos_tag2ids["O"]] * diff_len
            if 10 < len(conv_pos):
                conv_pos = conv_pos[:10]
            pos_ids.append(conv_pos)

        # Eojeol Boundary
        eojeol_ids: List[int] = []
        for mecab_item in new_mecab_item_list:
            token_size = 0
            for tok_pos_item in mecab_item.tok_pos_list:
                token_size += len(tok_pos_item.tokens)
            eojeol_ids.append(token_size)

        # 토큰 단위
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        if token_max_len <= len(text_tokens):
            text_tokens = text_tokens[:token_max_len-1]
            text_tokens.append("[SEP]")

            valid_token_len = token_max_len
        else:
            text_tokens.append("[SEP]")

            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (token_max_len - valid_token_len)

        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = ([1] * valid_token_len) + ([0] * (token_max_len - valid_token_len))
        token_type_ids = [0] * token_max_len

        # 어절 단위
        valid_eojeol_len = 0
        pos_ids.insert(0, [pos_tag2ids["O"]] * 10) # [CLS]
        if eojeol_max_len <= len(pos_ids):
            pos_ids = pos_ids[:eojeol_max_len-1]
            pos_ids.append([pos_tag2ids["O"]] * 10) # [SEP]
        else:
            pos_ids_size = len(pos_ids)
            for _ in range(eojeol_max_len - pos_ids_size):
                pos_ids.append([pos_tag2ids["O"]] * 10)

        # label_ids
        label_ids.insert(0, ETRI_TAG["O"])
        if eojeol_max_len <= len(label_ids):
            label_ids = label_ids[:eojeol_max_len-1]
            label_ids.append(ETRI_TAG["O"])
            valid_eojeol_len = eojeol_max_len
        else:
            label_ids_size = len(label_ids)
            valid_eojeol_len = label_ids_size
            for _ in range(eojeol_max_len - label_ids_size):
                label_ids.append(ETRI_TAG["O"])

        # eojeol_ids
        eojeol_ids.insert(0, 1) # [CLS]
        if eojeol_max_len <= len(eojeol_ids):
            eojeol_ids = eojeol_ids[:eojeol_max_len-1]
            eojeol_ids.append(1) # [SEP]
        else:
            eojeol_ids_size = len(eojeol_ids)
            eojeol_ids += [0] * (eojeol_max_len - eojeol_ids_size)

        # Check Size
        assert len(input_ids) == token_max_len, f"{len(input_ids)} + {input_ids}"
        assert len(attention_mask) == token_max_len, f"{len(attention_mask)} + {attention_mask}"
        assert len(token_type_ids) == token_max_len, f"{len(token_type_ids)} + {token_type_ids}"
        assert len(label_ids) == eojeol_max_len, f"{len(label_ids)} + {label_ids}"
        assert len(pos_ids) == eojeol_max_len, f"{len(pos_ids)} + {pos_ids}"
        assert len(eojeol_ids) == eojeol_max_len, f"{len(eojeol_ids)} + {eojeol_ids}"

        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["labels"].append(label_ids)

        # convert tag
        pos_ids = convert_pos_tag_to_combi_tag(pos_ids, use_nikl=False)
        npy_dict["pos_tag_ids"].append(pos_ids)
        npy_dict["eojeol_ids"].append(eojeol_ids)

        debug_pos_tag_ids = np.array(pos_ids)
        if "int32" != debug_pos_tag_ids.dtype:
            print(debug_pos_tag_ids)
            input()
        # debug_mode
        if debug_mode:
            print(src_item.text)
            print("Text Tokens: \n", text_tokens)
            print("new_mecab_item_list: \n", new_mecab_item_list)
            print("NE List: \n", src_item.ne_list)
            debug_pos_tag_ids = [[pos_ids2tag[x] for x in pos_tag_item] for pos_tag_item in pos_ids]
            for mecab_item, ne_lab, pos, ej_id in zip(new_mecab_item_list, label_ids[1:], debug_pos_tag_ids[1:], eojeol_ids[1:]):
                conv_tokens = [x.tokens for x in mecab_item.tok_pos_list]
                print(conv_tokens, ne_ids2tag[ne_lab], pos, ej_id)
            input()

    save_mecab_eojeol_npy(npy_dict, len(src_list), save_dir=save_model_dir)

#==========================================================================================
def save_mecab_eojeol_npy(npy_dict: Dict[str, List], src_list_len, save_dir: str = None):
#==========================================================================================
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["pos_tag_ids"] = np.array(npy_dict["pos_tag_ids"])
    npy_dict["eojeol_ids"] = np.array(npy_dict["eojeol_ids"])

    # wordpiece 토큰 단위
    print(f"Unit: Tokens")
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")

    # 어절 단위
    print(f"Unit: Eojoel")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")
    print(f"eojeol_ids.shape: {npy_dict['eojeol_ids'].shape}")

    # train/dev/test 분할
    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    # Train
    train_np = [npy_dict["input_ids"][:train_size],
                npy_dict["attention_mask"][:train_size],
                npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_labels_np = npy_dict["labels"][:train_size]
    train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]
    train_eojeol_ids_np = npy_dict["eojeol_ids"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_labels_np.shape: {train_labels_np.shape}")
    print(f"train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")
    print(f"train_eojeol_ids_np.shape: {train_eojeol_ids_np.shape}")

    # Dev
    dev_np = [npy_dict["input_ids"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size],
              npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_labels_np = npy_dict["labels"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]
    dev_eojeol_ids_np = npy_dict["eojeol_ids"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_labels_np.shape: {dev_labels_np.shape}")
    print(f"dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")
    print(f"dev_eojeol_ids_np.shape: {dev_eojeol_ids_np.shape}")


    # Test
    test_np = [npy_dict["input_ids"][valid_size:],
               npy_dict["attention_mask"][valid_size:],
               npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_labels_np = npy_dict["labels"][valid_size:]
    test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]
    test_eojeol_ids_np = npy_dict["eojeol_ids"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_labels_np.shape: {test_labels_np.shape}")
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

    # save pos_tag_ids
    np.save(root_path + "/train_pos_tag", train_pos_tag_np)
    np.save(root_path + "/dev_pos_tag", dev_pos_tag_np)
    np.save(root_path + "/test_pos_tag", test_pos_tag_np)

    # save eojeol_ids
    np.save(root_path+"/train_eojeol_ids", train_eojeol_ids_np)
    np.save(root_path+"/dev_eojeol_ids", dev_eojeol_ids_np)
    np.save(root_path+"/test_eojeol_ids", test_eojeol_ids_np)

    print(f"[make_gold_corpus_npy][save_eojeol_npy_dict] Complete - Save all npy files !")

#==========================================================================================
def mecab_token_unk_count(src_path: str = ""):
#==========================================================================================
    print("target_path: ", src_path)

    input_ids_np = np.load(src_path)
    input_ids_np = input_ids_np[:, :, 0]
    total_token_cnt = 0
    unk_cnt = 0
    for input_row in input_ids_np:
        token_count = np.where(input_row != 0)[0]
        token_count = len(token_count) - 2 # [CLS], [SEP]
        total_token_cnt += token_count
        unk_filter = np.where(input_row == 1)[0]
        if 0 < len(unk_filter):
            unk_cnt += len(unk_filter)
    print(f"total_token_cnt: {total_token_cnt}, mean_token_count: {total_token_cnt/input_ids_np.shape[0]}, "
          f"unk_cnt: {unk_cnt} / {unk_cnt/total_token_cnt*100}%")

#==========================================================================================
def mecab_pos_unk_count(src_sent_list: str = ""):
#==========================================================================================
    print(f"[mecab_pos_unk_count] src_sent_list.len: {len(src_sent_list)}")

    mecab = Mecab()
    mecab_tag_sets = [v for k, v in MECAB_POS_TAG.items()]
    print(f"mecab_pos_tag_sets: {mecab_tag_sets}")

    random.shuffle(src_sent_list)

    train_dev_size = int(len(src_sent_list) * 0.8)
    print(f"train_dev_size: {train_dev_size}")
    src_sent_list = src_sent_list[:train_dev_size]
    total_morp_cnt = 0
    total_pos_cnt = 0
    total_unk_pos_cnt = 0
    unk_pos_dict = {}
    for src_sent in src_sent_list:
        mecab_res = mecab.pos(src_sent.text)
        for word, pos in mecab_res:
            total_morp_cnt += 1

            split_pos = pos.split("+")
            total_pos_cnt += len(split_pos)
            for sp_p in split_pos:
                if sp_p not in mecab_tag_sets:
                    total_unk_pos_cnt += 1
                    if sp_p not in unk_pos_dict.keys():
                        unk_pos_dict[sp_p] = [word]
                    else:
                        unk_pos_dict[sp_p].append(word)
    print(f"Complete - total_morp_cnt: {total_morp_cnt}, total_pos_cnt: {total_pos_cnt}, "
          f"total_unk_pos_cnt: {total_unk_pos_cnt}, unk_pos/morp: {total_unk_pos_cnt/total_morp_cnt * 100}")
    print(unk_pos_dict)

#==========================================================================================
def make_mecab_wordpiece_npy(
        tokenizer_name: str, src_list: List[Sentence], target_n_pos: int,
        token_max_len: int = 128, debug_mode: bool = False,
        save_model_dir: str = None,
):
#==========================================================================================
    print("[make_mecab_wordpiece_npy] START !, src_list.len: ", len(src_list))

    # shuffle
    random.shuffle(src_list)

    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "pos_ids": [],
        "pos_flag": []
    }

    # Init
    pos_tag2ids = {v: k for k, v in MECAB_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in MECAB_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}
    ne_tag2ids = {k: v for k, v in ETRI_TAG.items()}

    mecab = Mecab()
    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    total_data_count = 0
    for proc_idx, src_item in enumerate(src_list):
        if debug_mode:
            is_test_str = False
            for test_str in test_str_list:
                if test_str in src_item.text:
                    is_test_str = True
            if not is_test_str:
                continue

        # 데이터 필터링
        if 0 == len(src_item.ne_list) or 3 >= len(src_item.word_list):
            continue

        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")
        total_data_count += 1

        # Mecab
        res_mecab = mecab.pos(src_item.text)
        # [['본래', ['NNG']], ['이날', ['NNG']], ['강릉', ['NNP']], ['에서', ['JKB']], ['선거', ['NNG']]
        conv_mecab_res = convert_wordpiece_pos_tokens(res_mecab, src_item.text)

        text_tokens = tokenizer.tokenize(src_item.text)
        text_tokens.insert(0, "[CLS]")

        # pos_ids
        mecab_tag2id = {v: k for k, v in MECAB_POS_TAG.items()}
        mecab_id2tag = {k: v for k, v in MECAB_POS_TAG.items()}
        pos_ids = [[mecab_tag2id["O"]] for _ in range(len(text_tokens))]
        pos_ids[0] = [mecab_tag2id["O"]] # [CLS]
        b_check_pos_use = [False for _ in range(len(text_tokens))]
        for tok_pos in conv_mecab_res:
            curr_pos = []
            for pos in tok_pos[1]:
                filter_pos = pos if "UNKNOWN" != pos and "NA" != pos and "UNA" != pos and "VSV" != pos else "O"
                curr_pos.append(mecab_tag2id[filter_pos])

            for root_idx in range(1, len(text_tokens)):
                if b_check_pos_use[root_idx]:
                    continue
                concat_item_list = []
                for tok_idx in range(root_idx, len(text_tokens)):
                    if b_check_pos_use[tok_idx]:
                        continue
                    for sub_idx in range(tok_idx, len(text_tokens)):
                        concat_word = ["".join(x).replace("##", "") for x in text_tokens[tok_idx:sub_idx + 1]]
                        concat_item_list.append(("".join(concat_word), (tok_idx, sub_idx)))
                concat_item_list = [x for x in concat_item_list if tok_pos[0] in x[0]]
                if 0 >= len(concat_item_list):
                    continue
                concat_item_list.sort(key=lambda x: len(x[0]))
                target_idx_pair = concat_item_list[0][1]
                b_check_pos_use[target_idx_pair[0]] = b_check_pos_use[target_idx_pair[1]] = True
                # print(concat_item_list[0][0], tok_pos[0], tok_pos[1])
                if concat_item_list[0][0] == tok_pos[0]:
                    pos_ids[target_idx_pair[0]] = curr_pos
                break
        # end loop, pos_ids

        # POS bit flag
        pos_bit_flags = []
        for pos in pos_ids:
            curr_token_bit_flag = [0 for _ in range(target_n_pos)]
            for p_id in pos:
                if 0 == p_id:
                    continue
                conv_flag_idx = conv_mecab_pos_groping_index(pos_ids2tag[p_id])
                if not conv_flag_idx:
                    continue
                elif 8 == conv_flag_idx:
                    josa_ids = conv_mecab_josa_index(pos_ids2tag[p_id])
                    curr_token_bit_flag[conv_flag_idx] = josa_ids
                else:
                    curr_token_bit_flag[conv_flag_idx] = 1
            pos_bit_flags.append(curr_token_bit_flag)
        # end loop, pos_bit_flag

        ''' For POS Embedding '''
        max_pos_size = 10
        for pos in pos_ids:
            if max_pos_size < len(pos):
                pos = pos[:max_pos_size]
            else:
                curr_len = len(pos)
                pos += [pos_tag2ids["O"]] * (max_pos_size - curr_len)

        # TEST
        # for t, p_b, p_i in zip(text_tokens, pos_bit_flags, pos_ids):
        #     print(t, p_b, p_i)
        # input()

        # Make NE Labels
        token_pair_len = len(text_tokens)
        label_ids = [ETRI_TAG["O"]] * token_pair_len
        b_check_use = [False for _ in range(token_pair_len)]
        for ne_idx, ne_item in enumerate(src_item.ne_list):
            ne_char_list = list(ne_item.text.replace(" ", ""))
            concat_item_list = []
            for m_idx, mecab_item in enumerate(text_tokens):
                if b_check_use[m_idx]:
                    continue
                for sub_idx in range(m_idx + 1, len(text_tokens)):
                    concat_word = ["".join(x).replace("##", "") for x in text_tokens[m_idx:sub_idx]]
                    concat_item_list.append(("".join(concat_word), (m_idx, sub_idx)))
            concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
            concat_item_list.sort(key=lambda x: len(x[0]))
            '''
                @TODO: 가장 좋은 TEST Case에서 완전 일치 아니면 O하는 걸로 해보면?
                근데 어차피 같은 형태의 개체명이면 동일하게 짤릴거 같은데
            '''
            if 0 >= len(concat_item_list):
                continue
            target_idx_pair = concat_item_list[0][1]

            for bio_idx in range(target_idx_pair[0], target_idx_pair[1]):
                b_check_use[bio_idx] = True
                if bio_idx == target_idx_pair[0]:
                    label_ids[bio_idx] = ETRI_TAG["B-" + ne_item.type]
                else:
                    label_ids[bio_idx] = ETRI_TAG["I-" + ne_item.type]
        # end, Make NE Labels
        # for t, l in zip(text_tokens, label_ids):
        #     print(t, l)
        # input()

        # 길이 맞춰주기
        valid_token_len = 0
        if token_max_len <= len(text_tokens):
            text_tokens = text_tokens[:token_max_len - 1]
            text_tokens.append("[SEP]")
            valid_token_len = token_max_len
        else:
            text_tokens.append("[SEP]")
            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (token_max_len - valid_token_len)

        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = ([1] * valid_token_len) + ([0] * (token_max_len - valid_token_len))
        token_type_ids = [0] * token_max_len

        if token_max_len <= len(pos_ids):
            pos_ids = pos_ids[:token_max_len - 1]
            pos_ids.append([0 for _ in range(max_pos_size)]) # SEP
        else:
            pos_ids_size = len(pos_ids)
            pos_ids += [[0 for _ in range(max_pos_size)]] * (token_max_len - pos_ids_size)

        if token_max_len <= len(pos_bit_flags):
            pos_bit_flags = pos_bit_flags[:token_max_len - 1]
            pos_bit_flags.append([0 for _ in range(target_n_pos)])  # [SEP]
        else:
            pos_ids_size = len(pos_bit_flags)
            pos_bit_flags += [[0 for _ in range(target_n_pos)]] * (token_max_len - pos_ids_size)

        if token_max_len <= len(label_ids):
            label_ids = label_ids[:token_max_len - 1]
            label_ids.append(ETRI_TAG["O"])
        else:
            label_ids_size = len(label_ids)
            label_ids += [ETRI_TAG["O"]] * (token_max_len - label_ids_size)

        assert len(input_ids) == token_max_len, f"{len(input_ids)}"
        assert len(attention_mask) == token_max_len, f"{len(attention_mask)}"
        assert len(token_type_ids) == token_max_len, f"{len(token_type_ids)}"
        assert len(label_ids) == token_max_len, f"{len(label_ids)}"
        assert len(pos_ids) == token_max_len, f"{len(pos_ids)}"
        assert len(pos_bit_flags) == token_max_len, f"{len(pos_bit_flags)}"

        # Insert npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["labels"].append(label_ids)
        npy_dict["pos_ids"].append(pos_ids)
        npy_dict["pos_flag"].append(pos_bit_flags)

        # Debug mode
        if debug_mode:
            print(src_item.text)
            print("Wordpiece: \n", tokenizer.tokenize(src_item.text))
            print("Text tokens: \n", text_tokens)
            print("NE labels: \n", src_item.ne_list)

            # Token 별 확인
            for t_tok, lab, p in zip(text_tokens, label_ids, pos_ids):
                if "[PAD]" == t_tok:
                    break
                print(t_tok, ne_ids2tag[lab], p)
            input()

    if not debug_mode:
        save_mecab_morp_npy(npy_dict, src_list_len=total_data_count, save_dir=save_model_dir)

#==========================================================================================
def conv_mecab_pos_groping_index(origin_pos: str):
#==========================================================================================
    ret_conv_idx = None
    if origin_pos in ["NNG", "NNP"]: # 일반 명사, 고유 명사
        ret_conv_idx = 0
    elif origin_pos in ["NNB"]: # 의존 명사
        ret_conv_idx = 1
    elif origin_pos in ["NNBC"]: # 단위를 나타내는 명사
        ret_conv_idx = 2
    elif origin_pos in ["SN"]: # 숫자
        ret_conv_idx = 3
    elif origin_pos in ["VV"]: # 동사
        ret_conv_idx = 4
    elif origin_pos in ["VA"]: # 형용사
        ret_conv_idx = 5
    elif origin_pos in ["JX"]: # 보조사
        ret_conv_idx = 6
    elif origin_pos in ["MM", "MAG"]: # 관형사, 일반 부사
        ret_conv_idx = 7
    elif origin_pos in ["JKS", "JKG", "JKO", "JKB", "JC"]:
        # 주격 조사, 관형격 조사, 목적격 조사, 부사격 조사, 접속 조사
        ret_conv_idx = 8
    elif origin_pos in ["XSN"]: # 명사 파생 접미사
        ret_conv_idx = 9
    elif origin_pos in ["EC", "EP", "EF", "XSV", "XSA", "ETM", "ETN", "VX"]:
        # 연결 어미, 선어말 어미, 종결 어미, 동사 파생 접미사, 형용사 파생 접미사, 관형형 전성어미, 명사형 전성어미, 보조 용언
        ret_conv_idx = 10
    elif origin_pos in ["SSO", "SSC", "SY", "SC"]:
        # 여는 괄호, 닫는 괄호, (붙임표, 기타기호), 구분자
        ret_conv_idx = 11

    return ret_conv_idx

#==========================================================================================
def conv_mecab_josa_index(origin_pos: str):
#==========================================================================================
    josa_ids = None
    if "JKS" == origin_pos: # 주격 조사
        josa_ids = 1
    elif "JKG" == origin_pos: # 관형격 조사
        josa_ids = 2
    elif "JKO" == origin_pos: # 목적격 조사
        josa_ids = 3
    elif "JKB" == origin_pos: # 부사격 조사
        josa_ids = 4
    elif "JKC" == origin_pos: # 보격 조사
        josa_ids = 5
    elif "JC" == origin_pos:
        josa_ids = 6

    return josa_ids

#==========================================================================================
def save_mecab_wordpiece_npy(npy_dict: Dict[str, List], src_list_len, save_dir: str = None):
#==========================================================================================
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])

    npy_dict["pos_ids"] = np.array(npy_dict["pos_ids"])

    # 토큰 단위
    print(f"Unit: Tokens")
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"sents.len: {len(npy_dict['sentences'])}")
    # print(f"pos_tag_ids.shape: {npy_dict['pos_tag_ids'].shape}")
    # print(f"ne_one_hot.shape: {npy_dict['ne_one_hot'].shape}")
    # print(f"josa_one_hot.shape: {npy_dict['josa_one_hot'].shape}")

    # train/dev/test 분할
    split_size = int(src_list_len * 0.1)
    train_size = split_size * 7
    valid_size = train_size + split_size

    # Train
    train_np = [npy_dict["input_ids"][:train_size],
                npy_dict["attention_mask"][:train_size],
                npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_labels_np = npy_dict["labels"][:train_size]
    train_sents_list = npy_dict["sentences"][:train_size]
    # train_pos_tag_np = npy_dict["pos_tag_ids"][:train_size]
    # train_ne_one_hot_np = npy_dict["ne_one_hot"][:train_size]
    # train_josa_one_hot_np = npy_dict["josa_one_hot"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_labels_np.shape: {train_labels_np.shape}")
    print(f"train_sents_list.shape: {len(train_sents_list)}")
    # print(f"train_pos_tag_ids_np.shape: {train_pos_tag_np.shape}")
    # print(f"train_ne_one_hot_np.shape: {train_ne_one_hot_np.shape}")
    # print(f"train_josa_one_hot_np.shape: {train_josa_one_hot_np.shape}")

    # Dev
    dev_np = [npy_dict["input_ids"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size],
              npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_labels_np = npy_dict["labels"][train_size:valid_size]
    dev_sents_list = npy_dict["sentences"][train_size:valid_size]
    # dev_pos_tag_np = npy_dict["pos_tag_ids"][train_size:valid_size]
    # dev_ne_one_hot_np = npy_dict["ne_one_hot"][train_size:valid_size]
    # dev_josa_one_hot_np = npy_dict["josa_one_hot"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_labels_np.shape: {dev_labels_np.shape}")
    print(f"dev_sents_list.shape: {len(dev_sents_list)}")
    # print(f"dev_pos_tag_ids_np.shape: {dev_pos_tag_np.shape}")
    # print(f"dev_ne_one_hot_np.shape: {dev_ne_one_hot_np.shape}")
    # print(f"dev_josa_one_hot_np.shape: {dev_josa_one_hot_np.shape}")

    # Test
    test_np = [npy_dict["input_ids"][valid_size:],
               npy_dict["attention_mask"][valid_size:],
               npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_labels_np = npy_dict["labels"][valid_size:]
    test_sents_list = npy_dict["sentences"][valid_size:]
    # test_pos_tag_np = npy_dict["pos_tag_ids"][valid_size:]
    # test_ne_one_hot_np = npy_dict["ne_one_hot"][valid_size:]
    # test_josa_one_hot_np = npy_dict["josa_one_hot"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_labels_np.shape: {test_labels_np.shape}")
    print(f"test_sents_list.shape: {len(test_sents_list)}")
    # print(f"test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")
    # print(f"test_ne_one_hot_np.shape: {test_ne_one_hot_np.shape}")
    # print(f"test_josa_one_hot_np.shape: {test_josa_one_hot_np.shape}")

    root_path = "../corpus/npy/" + save_dir
    # save input_ids, attention_mask, token_type_ids
    np.save(root_path+"/train", train_np)
    np.save(root_path+"/dev", dev_np)
    np.save(root_path+"/test", test_np)

    # save labels
    np.save(root_path+"/train_labels", train_labels_np)
    np.save(root_path+"/dev_labels", dev_labels_np)
    np.save(root_path+"/test_labels", test_labels_np)

    # save pos_tag_ids
    # np.save(root_path + "/train_pos_tag", train_pos_tag_np)
    # np.save(root_path + "/dev_pos_tag", dev_pos_tag_np)
    # np.save(root_path + "/test_pos_tag", test_pos_tag_np)

    # save NE, Josa One-hot
    # np.save(root_path + "/train_ne_one_hot", train_ne_one_hot_np)
    # np.save(root_path + "/dev_ne_one_hot", dev_ne_one_hot_np)
    # np.save(root_path + "/test_ne_one_hot", test_ne_one_hot_np)

    # np.save(root_path + "/train_josa_one_hot", train_josa_one_hot_np)
    # np.save(root_path + "/dev_josa_one_hot", dev_josa_one_hot_np)
    # np.save(root_path + "/test_josa_one_hot", test_josa_one_hot_np)

    # Save Sentences - pickle
    with open(root_path + "/train_sents.pkl", mode="wb") as train_sents_pkl:
        pickle.dump(train_sents_list, train_sents_pkl)
    with open(root_path + "/dev_sents.pkl", mode="wb") as dev_sents_pkl:
        pickle.dump(dev_sents_list, dev_sents_pkl)
    with open(root_path + "test_sents.pkl", mode="wb") as test_sents_pkl:
        pickle.dump(test_sents_list, test_sents_pkl)

    print(f"Complete - Save all npy files !")

#==========================================================================================
def save_mecab_morp_npy(npy_dict: Dict[str, List], src_list_len, save_dir: str = None):
#==========================================================================================
    npy_dict["input_ids"] = np.array(npy_dict["input_ids"])
    npy_dict["attention_mask"] = np.array(npy_dict["attention_mask"])
    npy_dict["token_type_ids"] = np.array(npy_dict["token_type_ids"])
    npy_dict["labels"] = np.array(npy_dict["labels"])
    npy_dict["pos_ids"] = np.array(npy_dict["pos_ids"])
    npy_dict["pos_flag"] = np.array(npy_dict["pos_flag"])

    # 토큰 단위
    print(f"Unit: Tokens")
    print(f"input_ids.shape: {npy_dict['input_ids'].shape}")
    print(f"attention_mask.shape: {npy_dict['attention_mask'].shape}")
    print(f"token_type_ids.shape: {npy_dict['token_type_ids'].shape}")
    print(f"labels.shape: {npy_dict['labels'].shape}")
    print(f"pos_ids.shape: {npy_dict['pos_ids'].shape}")
    print(f"pos_flag.shape: {npy_dict['pos_flag'].shape}")

    # train/dev/test 분할
    split_size = int(src_list_len * 0.1)
    train_size = split_size * 8
    valid_size = train_size + split_size
    print(f"split_size: {split_size}, train_size: {train_size}, valid_size: {valid_size}")

    # Train
    train_np = [npy_dict["input_ids"][:train_size],
                npy_dict["attention_mask"][:train_size],
                npy_dict["token_type_ids"][:train_size]]
    train_np = np.stack(train_np, axis=-1)
    train_labels_np = npy_dict["labels"][:train_size]
    train_pos_tag_np = npy_dict["pos_ids"][:train_size]
    train_pos_flag_np = npy_dict["pos_flag"][:train_size]
    print(f"train_np.shape: {train_np.shape}")
    print(f"train_labels_np.shape: {train_labels_np.shape}")
    print(f"train_pos_ids_np.shape: {train_pos_tag_np.shape}")
    print(f"train_pos_flag_np.shape: {train_pos_flag_np.shape}")

    # Dev
    dev_np = [npy_dict["input_ids"][train_size:valid_size],
              npy_dict["attention_mask"][train_size:valid_size],
              npy_dict["token_type_ids"][train_size:valid_size]]
    dev_np = np.stack(dev_np, axis=-1)
    dev_labels_np = npy_dict["labels"][train_size:valid_size]
    dev_pos_tag_np = npy_dict["pos_ids"][train_size:valid_size]
    dev_pos_flag_np = npy_dict["pos_flag"][train_size:valid_size]
    print(f"dev_np.shape: {dev_np.shape}")
    print(f"dev_labels_np.shape: {dev_labels_np.shape}")
    print(f"dev_pos_ids_np.shape: {dev_pos_tag_np.shape}")
    print(f"dev_pos_flag_np.shape: {dev_pos_flag_np.shape}")

    # Test
    test_np = [npy_dict["input_ids"][valid_size:],
               npy_dict["attention_mask"][valid_size:],
               npy_dict["token_type_ids"][valid_size:]]
    test_np = np.stack(test_np, axis=-1)
    test_labels_np = npy_dict["labels"][valid_size:]
    test_pos_tag_np = npy_dict["pos_ids"][valid_size:]
    test_pos_flag_np = npy_dict["pos_flag"][valid_size:]
    print(f"test_np.shape: {test_np.shape}")
    print(f"test_labels_np.shape: {test_labels_np.shape}")
    print(f"test_pos_tag_ids_np.shape: {test_pos_tag_np.shape}")
    print(f"test_pos_flag_np.shape: {test_pos_flag_np.shape}")

    root_path = "../corpus/npy/" + save_dir
    # save input_ids, attention_mask, token_type_ids
    np.save(root_path + "/train", train_np)
    np.save(root_path + "/dev", dev_np)
    np.save(root_path + "/test", test_np)

    # save labels
    np.save(root_path + "/train_label_ids", train_labels_np)
    np.save(root_path + "/dev_label_ids", dev_labels_np)
    np.save(root_path + "/test_label_ids", test_labels_np)

    np.save(root_path + "/train_pos_ids", train_pos_tag_np)
    np.save(root_path + "/dev_pos_ids", dev_pos_tag_np)
    np.save(root_path + "/test_pos_ids", test_pos_tag_np)

    np.save(root_path + "/train_pos_flag", train_pos_flag_np)
    np.save(root_path + "/dev_pos_flag", dev_pos_flag_np)
    np.save(root_path + "/test_pos_flag", test_pos_flag_np)

    print(f"Complete - Save all npy files !")

#================================================================
@dataclass
class Morp_pair:
    morp: List[str] = field(default_factory=list)
    pos: List[str] = field(default_factory=list)

@dataclass
class Compare_res:
    sent_id: str = ""
    sent_text: str = ""
    target: Tuple[int, Tuple[str, str]] = field(default_factory=tuple)
    nikl_morp: List[str] = field(default_factory=list)
    mecab_morp: List[str] = field(default_factory=list)

#==========================================================================================
def conv_nikl_pos_to_mecab(nikl_morp):
#==========================================================================================
    MM_list = ["MMA", "MMD", "MMN"]
    SP_list = ["SO", "SW"]
    ret_morp = nikl_morp
    if nikl_morp in MM_list:
        ret_morp = "MM"
    # elif nikl_morp in SP_list:
    #     ret_morp = "SY"

    return ret_morp

#==========================================================================================
def conv_mecab_pos_to_nikl(mecab_pos_list):
#==========================================================================================
    ret_conv_pos_list = []
    for mecab_pos in mecab_pos_list:
        if "NNBC" == mecab_pos:
            ret_conv_pos_list.append("NNB")
        else:
            ret_conv_pos_list.append(mecab_pos)
    return ret_conv_pos_list

#==========================================================================================
def compare_mecab_and_gold_corpus(src_corpus_list: List[Sentence] = []):
#==========================================================================================
    '''
        cmp_results_dict = { key: POS, value: { [(sent_id, sent_text, [(word_id, gold_corpus, mecab_pos_concat]] }
    '''
    cmp_results_dict = {}

    # Mecab
    mecab = Mecab()
    for sent_item in src_corpus_list:
        # Gold Corpus
        gold_corpus_dict = {} # key: word_id, value = Word_POS_pair(word, pos)
        # Set word_id(key)
        for word_item in sent_item.word_list:
            gold_corpus_dict[word_item.id] = []

        # Set NIKL POS
        for key in gold_corpus_dict.keys():
            pos_list = []
            form_list = []
            for morp_item in sent_item.morp_list:
                if key == morp_item.word_id:
                    # word
                    form_list.append(morp_item.form)
                    # morp
                    conv_label = conv_nikl_pos_to_mecab(nikl_morp=morp_item.label)
                    pos_list.append(conv_label)
            morp_pair = Morp_pair(morp=form_list, pos=pos_list)
            gold_corpus_dict[key] = morp_pair

        # Set Mecab POS
        mecab_res_dict = {} # key: word_id, value = Word_POS_pair(word, pos)
        mecab_res_deque = deque()
        for mecab_res in mecab.pos(sent_item.text):
            mecab_res_deque.append([mecab_res[0], mecab_res[1]])

        eojeol_list = sent_item.text.split()
        for e_idx, eojeol in enumerate(eojeol_list):
            target_list = []
            while 0 != len(mecab_res_deque):
                left_item = mecab_res_deque.popleft()
                if left_item[0] in eojeol:
                    target_list.append(left_item)
                else:
                    mecab_res_deque.appendleft(left_item)
                    break
            concat_word = [x[0] for x in target_list]
            concat_pos = []
            for x in target_list:
                sp_pos = x[1].split("+")
                sp_pos = conv_mecab_pos_to_nikl(sp_pos)
                concat_pos.extend(sp_pos)
            word_pos_pair = Morp_pair(morp=concat_word, pos=concat_pos)
            mecab_res_dict[e_idx+1] = word_pos_pair

        # Compare
        ignore_list = ["SF", "SE", "SS", "SP", "SO", "SW", "SSO", "SSC", "SC", "SY"] # "SL", "SH", "SN"
        for key, value in gold_corpus_dict.items():
            conv_nikl_giho_value = ["GH" if x in ignore_list else x for x in value.pos]

            mecab_dict_value = mecab_res_dict[key]
            conv_mecab_giho_value = ["GH" if x in ignore_list else x for x in mecab_dict_value.pos]
            if "+".join(conv_nikl_giho_value) != "+".join(conv_mecab_giho_value):
                # 형태소 분석 정보의 길이가 같다
                if len(conv_nikl_giho_value) == len(conv_mecab_giho_value):
                    for idx, (nikl_pos, mecab_pos) in enumerate(zip(conv_nikl_giho_value, conv_mecab_giho_value)):
                        if nikl_pos != mecab_pos:
                            filtered_target = (idx, (value.morp[idx], value.pos[idx]))
                            cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                                  target=filtered_target,
                                                  nikl_morp=value, mecab_morp=mecab_dict_value)
                            if nikl_pos not in cmp_results_dict.keys():
                                cmp_results_dict[nikl_pos] = [cmp_res]
                            else:
                                cmp_results_dict[nikl_pos].append(cmp_res)
                # gold corpus의 형태소 분석 정보가 더 많다.
                elif len(conv_nikl_giho_value) > len(conv_mecab_giho_value):
                    filtered_idx = -1
                    for idx, mecab_pos in enumerate(conv_mecab_giho_value):
                        if mecab_pos != conv_nikl_giho_value[idx]:
                            filtered_idx = idx
                            break
                    filtered_target = ()
                    cmp_res = Compare_res()
                    if -1 != filtered_idx:
                        filtered_target = (filtered_idx, (value.morp[filtered_idx], value.pos[filtered_idx]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target,
                                              nikl_morp=value, mecab_morp=mecab_dict_value)
                    else:
                        filtered_target = (0, (value.morp[0], value.pos[0]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target, nikl_morp=value, mecab_morp=mecab_dict_value)
                    if value.pos[filtered_idx] not in cmp_results_dict.keys():
                        cmp_results_dict[value.pos[filtered_idx]] = [cmp_res]
                    else:
                        cmp_results_dict[value.pos[filtered_idx]].append(cmp_res)
                # mecab의 형태소 분석 정보가 더 많다.
                else:
                    filtered_idx = -1
                    for idx, nikl_pos in enumerate(conv_nikl_giho_value):
                        if nikl_pos != conv_mecab_giho_value[idx]:
                            filtered_idx = idx
                            break
                    filtered_target = ()
                    cmp_res = Compare_res()
                    if -1 != filtered_idx:
                        filtered_target = (filtered_idx, (value.morp[filtered_idx], value.pos[filtered_idx]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target,
                                              nikl_morp=value, mecab_morp=mecab_dict_value)
                    else:
                        filtered_target = (0, (value.morp[0], value.pos[0]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target,
                                              nikl_morp=value, mecab_morp=mecab_dict_value)
                    if value.pos[filtered_idx] not in cmp_results_dict.keys():
                        cmp_results_dict[value.pos[filtered_idx]] = [cmp_res]
                    else:
                        cmp_results_dict[value.pos[filtered_idx]].append(cmp_res)

    # Complete compare
    for key, value in cmp_results_dict.items():
        with open("./mecab_cmp/"+key+".txt", mode="w", encoding="utf-8") as write_file:
            print(f"Value Size: {len(value)}")
            for v in value:
                write_file.write("sent_id: "+v.sent_id+"\n")
                write_file.write("sent_text: "+v.sent_text+"\n")
                write_file.write("target: "+str(v.target)+"\n")
                write_file.write("nikl_item: "+str(v.nikl_morp)+"\n")
                write_file.write("mecab_item: " + str(v.mecab_morp) + "\n")
                write_file.write("\n\n")

    # Save Dictionary
    with open("./mecab_cmp/mecab_compare_dict.pkl", mode="wb") as write_pkl:
        pickle.dump(cmp_results_dict, write_pkl)
        print("save pkl")

#==========================================================================================
def check_nikl_and_mecab_difference(dic_pkl_path: str = ""):
#==========================================================================================
    load_dict = {}
    with open(dic_pkl_path, mode="rb") as dict_pkl:
        load_dict = pickle.load(dict_pkl)

    # Check Total errors
    total_diff_cnt = 0
    for key, val in load_dict.items():
        total_diff_cnt += len(val)

    # Check (NNG to VV) or (NNP to VV)
    nng2vv_err = 0
    target_key = "NNG"
    search_target = "NNG"
    target_origin_dict_key: List[Compare_res] = load_dict[target_key]

    search_dict = {}
    err_cnt = 0
    for nng_val in target_origin_dict_key:
        target_idx = nng_val.target[0]
        mecab_morp_pos_pair = nng_val.mecab_morp
        if 0 >= len(mecab_morp_pos_pair.morp):
            continue
        try:
            mecab_morp = mecab_morp_pos_pair.morp[target_idx]
            mecab_pos = mecab_morp_pos_pair.pos[target_idx]
        except:
            print(target_idx)
            print(nng_val.sent_text)
            print(nng_val.target)
            print(mecab_morp_pos_pair.morp)
            print(mecab_morp_pos_pair.pos)
            err_cnt += 1
            print(err_cnt)

        if mecab_pos not in search_dict.keys():
            search_dict[mecab_pos] = [(nng_val.sent_text, nng_val.nikl_morp, mecab_morp_pos_pair)]
        else:
            search_dict[mecab_pos].append((nng_val.sent_text, nng_val.nikl_morp, mecab_morp_pos_pair))

    for sent, lhs, rhs in search_dict[search_target]:
        print("Sent: ", sent)
        print("NIKL: ", lhs)
        print("MECAB: ", rhs)
        print("===================")
    print(f"Target POS: {target_key}, Search Target: {search_target}, Size: {len(search_dict[search_target])}")

    print("====================\nsearch_dict diff info ")
    search_dict_rank = []
    for key in search_dict.keys():
        search_dict_rank.append((key, len(search_dict[key])))
    search_dict_rank = sorted(search_dict_rank, key=lambda x: x[1], reverse=True)
    for lhs, rhs in search_dict_rank:
        print(f"{lhs} Diff item size: {rhs}")

    print("====================\nsrc_dict diff info ")
    load_dict_rank = []
    for key in load_dict.keys():
        load_dict_rank.append((key, len(load_dict[key])))
    load_dict_rank = sorted(load_dict_rank, key=lambda x: x[1], reverse=True)
    for lhs, rhs in load_dict_rank:
        print(f"{lhs} Diff item size: {rhs}")
    print(f"====================\nTotal Diff count: {total_diff_cnt}")

#==========================================================================================
def check_count_morp(src_sent_list):
#==========================================================================================
    nikl_morp_count = 0
    mecab_morp_count = 0
    mecab_pos_count = 0

    mecab = Mecab()
    for sent_item in src_sent_list:
        mecab_res = mecab.pos(sent_item.text)
        for item in mecab_res:
            mecab_morp_count += 1
            mecab_pos_count += len(item[1])
        nikl_morp_count += len(sent_item.morp_list)
    print("nikl_morp_count: ", nikl_morp_count)
    print("mecab_morp_count: ", mecab_morp_count)
    print("mecab_pos_count: ", mecab_pos_count)

#==========================================================================================
def make_mecab_morp_npy(
        tokenizer_name: str, src_list: List[Sentence],
        token_max_len: int = 128, debug_mode: bool = False,
        save_model_dir: str = None,
        target_n_pos: int = 14, target_tag_list: List[str] = []
):
#==========================================================================================
    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "pos_ids": [],
        "pos_flag": []
    }

    # shuffle
    random.shuffle(src_list)

    # Init
    pos_tag2ids = {v: int(k) for k, v in MECAB_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in MECAB_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}

    mecab = Mecab()
    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    total_data_count = 0
    for proc_idx, src_item in enumerate(src_list):
        # Test
        if debug_mode:
            is_test_str = False
            for test_str in test_str_list:
                if test_str in src_item.text:
                    is_test_str = True
            if not is_test_str:
                continue

        if 0 == len(src_item.ne_list) and 3 >= len(src_item.word_list):
            continue

        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")
        total_data_count += 1

        # Mecab
        res_mecab = mecab.pos(src_item.text)
        # [('전창수', 'NNP', False), ('(', 'SSO', False), ('42', 'SN', False)]
        conv_mecab_res = convert_morp_connected_tokens(res_mecab, src_item.text)

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

        # Make POS
        pos_ids = []
        for tok_pos in token_pos_list:
            curr_pos = [pos_tag2ids["O"]] * 10
            for p_idx, pos in enumerate(tok_pos):
                if 10 <= p_idx:
                    continue
                filter_pos = pos if "UNKNOWN" != pos and "NA" != pos and "UNA" != pos and "VSV" != pos else "O"
                curr_pos[p_idx] = pos_tag2ids[filter_pos]
            pos_ids.append(curr_pos)

        # Make POS Flag
        mecab_tag2ids = {v: k for k, v in MECAB_POS_TAG.items()}  # origin, 1: "NNG"
        # {1: 0, 2: 1, 43: 2, 3: 3, 5: 4, 4: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, 23: 11, 24: 12, 21: 13, 22: 14}
        target_tag2ids = {mecab_tag2ids[x]: i for i, x in enumerate(target_tag_list)}
        pos_target_onehot = []
        for seq_pos in pos_ids:
            span_pos = [0 for _ in range(target_n_pos)]
            for pos_item in seq_pos:
                if pos_item in target_tag2ids.keys():
                    if 0 == pos_item or 1 == pos_item:
                        span_pos[0] = 1
                    else:
                        span_pos[target_tag2ids[pos_item] - 1] = 1
            pos_target_onehot.append(span_pos)

        # Make NE Labels
        token_pair_len = len(text_tokens)
        labels_ids = [ETRI_TAG["O"]] * token_pair_len
        b_check_use = [False for _ in range(token_pair_len)]
        for ne_idx, ne_item in enumerate(src_item.ne_list):
            ne_char_list = list(ne_item.text.replace(" ", ""))
            concat_item_list = []
            for m_idx, mecab_item in enumerate(text_tokens):
                if b_check_use[m_idx]:
                    continue
                for sub_idx in range(m_idx + 1, len(text_tokens)):
                    concat_word = ["".join(x).replace("##", "") for x in text_tokens[m_idx:sub_idx]]
                    concat_item_list.append(("".join(concat_word), (m_idx, sub_idx)))
            concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
            concat_item_list.sort(key=lambda x: len(x[0]))
            if 0 >= len(concat_item_list):
                continue
            target_idx_pair = concat_item_list[0][1]

            for bio_idx in range(target_idx_pair[0], target_idx_pair[1]):
                b_check_use[bio_idx] = True
                if bio_idx == target_idx_pair[0]:
                    labels_ids[bio_idx] = ETRI_TAG["B-" + ne_item.type]
                else:
                    labels_ids[bio_idx] = ETRI_TAG["I-" + ne_item.type]
        # end, Make NE Labels

        # Matching Length
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        if token_max_len <= len(text_tokens):
            text_tokens = text_tokens[:token_max_len - 1]
            text_tokens.append("[SEP]")

            valid_token_len = token_max_len
        else:
            text_tokens.append("[SEP]")

            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (token_max_len - valid_token_len)

        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = ([1] * valid_token_len) + ([0] * (token_max_len - valid_token_len))
        token_type_ids = [0] * token_max_len

        # POS
        pos_ids.insert(0, [pos_tag2ids["O"]] * 10)  # [CLS]
        if token_max_len <= len(pos_ids):
            pos_ids = pos_ids[:token_max_len - 1]
            pos_ids.append([pos_tag2ids["O"]] * 10)  # [SEP]
        else:
            pos_ids_size = len(pos_ids)
            for _ in range(token_max_len - pos_ids_size):
                pos_ids.append([pos_tag2ids["O"]] * 10)

        pos_target_onehot.insert(0, [0 for _ in range(target_n_pos)]) # [CLS]
        if token_max_len <= len(pos_target_onehot):
            pos_target_onehot = pos_target_onehot[:token_max_len - 1]
            pos_target_onehot.append([0 for _ in range(target_n_pos)])
        else:
            pos_target_onehot_size = len(pos_target_onehot)
            for _ in range(token_max_len - pos_target_onehot_size):
                pos_target_onehot.append([0 for _ in range(target_n_pos)])

        # NE Label
        labels_ids.insert(0, ETRI_TAG["O"])
        if token_max_len <= len(labels_ids):
            labels_ids = labels_ids[:token_max_len - 1]
            labels_ids.append(ETRI_TAG["O"])
        else:
            label_ids_size = len(labels_ids)
            for _ in range(token_max_len - label_ids_size):
                labels_ids.append(ETRI_TAG["O"])

        # Check size
        assert len(input_ids) == token_max_len, f"{len(input_ids)} + {input_ids}"
        assert len(attention_mask) == token_max_len, f"{len(attention_mask)} + {attention_mask}"
        assert len(token_type_ids) == token_max_len, f"{len(token_type_ids)} + {token_type_ids}"
        assert len(labels_ids) == token_max_len, f"{len(labels_ids)} + {labels_ids}"
        assert len(pos_ids) == token_max_len, f"{len(pos_ids)} + {pos_ids}"

        # Insert npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["labels"].append(labels_ids)
        npy_dict["pos_ids"].append(pos_ids)
        npy_dict["pos_flag"].append(pos_target_onehot)

        # Debug mode
        if debug_mode:
            print(src_item.text)
            print("Wordpiece: \n", tokenizer.tokenize(src_item.text))
            print("Text tokens: \n", text_tokens)
            print("NE labels: \n", src_item.ne_list)

            # Token 별 확인
            for t_tok, lab, p, p_oh in zip(text_tokens, labels_ids, pos_ids, pos_target_onehot):
                if "[PAD]" == t_tok:
                    break
                conv_p = [pos_ids2tag[x] for x in p]
                print(t_tok, ne_ids2tag[lab], conv_p, p_oh)
            input()

    if not debug_mode:
        # if is_make_pos_flag:
            # npy_dict["pos_ids"] = npy_dict["pos_flag"]
        save_mecab_morp_npy(npy_dict, src_list_len=total_data_count, save_dir=save_model_dir)

#======================================================
def kor_letter_from(letter):
#======================================================
    lastLetterInt = 15572643

    if not letter:
        return '가'

    a = letter
    b = a.encode('utf8')
    c = int(b.hex(), 16)

    if c == lastLetterInt:
        return False

    d = hex(c + 1)
    e = bytearray.fromhex(d[2:])

    flag = True
    while flag:
        try:
            r = e.decode('utf-8')
            flag = False
        except UnicodeDecodeError:
            c = c + 1
            d = hex(c)
            e = bytearray.fromhex(d[2:])

    return e.decode()

#======================================================
def split_hangul_components(all_hangul: List[str]):
#======================================================
    comp_list = []

    # 한글
    for char in all_hangul:
        ch_split = list(j2hcj(h2j(char)))
        comp_list.extend(ch_split)
    comp_list = list(set(comp_list))

    # 숫자
    for num in range(0, 10):
        comp_list.append(str(num))

    # 영어
    comp_list.extend(ascii_lowercase)

    # TODO: 특수 문자

    comp_list = sorted(comp_list)
    comp_list.insert(0, "<p>") # PAD
    comp_list.insert(1, "<u>") # UNK
    comp_list.insert(2, "<e>") # end of char
    comp_list.insert(3, "_") # space

    return comp_list

#==========================================================================================
def make_char_dictionary(all_sent: List[Sentence], save_path:str):
#==========================================================================================
    # 말뭉치 내에서 기호 얻기
    char_set = []
    for sent in all_sent:
        for ch in list(sent.text.replace(" ", "_")):
            unicode_name = unicodedata.name(ch)
            if re.search(r"[가-힣a-zA-Z0-9]+", ch) or unicode_name.startswith("CJK") \
                    or unicode_name.startswith("KATAKANA") or unicode_name.startswith("HANGUL") or \
                    unicode_name.startswith("SOFT") or unicode_name.startswith("HIRAGANA"):
                continue
            else:
                char_set.append(str(ch))

    # 초/중/종성
    all_hangul = []
    add_ch = ''
    while True:
        add_ch = kor_letter_from(add_ch)
        if add_ch is False:
            break
        all_hangul.append(add_ch)
    hangul_comp_list = split_hangul_components(all_hangul)
    char_set.extend(hangul_comp_list)
    char_set = sorted(list(set(char_set)))

    char_dict = {c: i for i, c in enumerate(char_set)}
    # print(char_dict)
    # print(len(char_dict))

    # Save
    with open(save_path, mode="wb") as dict_pkl:
        pickle.dump(char_dict, dict_pkl)

    return char_dict

#======================================================
def make_jamo_one_hot(vocab_dict: Dict[str, int],
                      vocab_size: int, sent: str = "",
                      seq_len: int = 128):
#======================================================
    split_by_char = list(sent)

    # 3: 초/중/종성
    # chars_one_hot = np.zeros((seq_len * 3, vocab_size))
    chars_one_hot = np.zeros((seq_len, 3))
    add_idx = 0
    for ch_idx, char in enumerate(split_by_char):
        if 128 <= ch_idx:
            break
        jamo = j2hcj(h2j(char))
        zeros_np = np.zeros((3, ))
        for jm_idx, jm in enumerate(jamo):
            if jm not in vocab_dict.keys():
                # zeros_np[jm_idx, vocab_dict["<u>"]] = 1
                zeros_np[jm_idx] = vocab_dict["<u>"]
            else:
                # zeros_np[jm_idx, vocab_dict[jm]] = 1
                zeros_np[jm_idx] = vocab_dict[jm]
        chars_one_hot[add_idx] = zeros_np
        add_idx += 1

    return chars_one_hot

#======================================================
def load_sentences_datasets(src_sents: List[Sentence]) -> Tuple[List[str], List[str], List[str]]:
#======================================================
    # Load
    print(f"[load_sentences_datasets] Load data size: {len(src_sents)}")

    total_sents = []
    for sent in src_sents:
        total_sents.append(sent.text.replace(" ", "_"))

    # Shuffle
    split_size = int(len(total_sents) * 0.1)
    train_idx = split_size * 7
    dev_idx = train_idx + split_size

    train_sents = total_sents[:train_idx]
    dev_sents = total_sents[train_idx:dev_idx]
    test_sents = total_sents[dev_idx:]
    print(f"[load_sentences_datasets] size - train: {len(train_sents)}, dev: {len(dev_sents)}, test: {len(test_sents)}")

    return train_sents, dev_sents, test_sents


#==========================================================================================
def make_mecab_char_npy(
        tokenizer_name: str, src_list: List[Sentence],
        token_max_len: int = 128, debug_mode: bool = False,
        save_model_dir: str = None,
        target_n_pos: int = 14, target_tag_list: List[str] = []
):
#==========================================================================================
    npy_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "token_type_ids": [],
        "pos_ids": [],
        "pos_flag": []
    }

    # shuffle
    random.shuffle(src_list)

    # Init
    pos_tag2ids = {v: k for k, v in MECAB_POS_TAG.items()}
    pos_ids2tag = {k: v for k, v in MECAB_POS_TAG.items()}
    ne_ids2tag = {v: k for k, v in ETRI_TAG.items()}
    ne_tag2ids = {k: v for k, v in ETRI_TAG.items()}

    mecab = Mecab()
    tokenizer = KoCharElectraTokenizer.from_pretrained(tokenizer_name)

    total_data_count = 0
    for proc_idx, src_item in enumerate(src_list):
        if debug_mode:
            is_test_str = False
            for test_str in test_str_list:
                if test_str in src_item.text:
                    is_test_str = True
            if not is_test_str:
                continue

        # 데이터 필터링
        if 0 == len(src_item.ne_list) and 3 >= len(src_item.word_list):
            continue

        if 0 == (proc_idx % 1000):
            print(f"{proc_idx} Processing... {src_item.text}")
        total_data_count += 1

        # MeCab
        res_mecab = mecab.pos(src_item.text)
        # [('전창수', 'NNP', False), ('(', 'SSO', False), ('42', 'SN', False)]
        conv_mecab_res = convert_character_pos_tokens(res_mecab, src_item.text)

        # Tokenize
        text_tokens = tokenizer.tokenize(src_item.text)  # 숫자도 다 분리됨
        text_tokens.insert(0, "[CLS]")

        # pos_ids
        ''' For POS Bit Flag '''
        pos_flag = [[0 for _ in range(target_n_pos)]] # [CLS]
        target_tag2ids = {pos_tag2ids[x]: i for i, x in enumerate(target_tag_list)}
        for tok_pos in conv_mecab_res:
            conv_mecab_pos = []
            for pos in tok_pos[1]:
                filter_pos = pos if "UNKNOWN" != pos and "NA" != pos and "UNA" != pos and "VSV" != pos else "O"
                conv_mecab_pos.append(filter_pos)
            for sy_idx, syllable in enumerate(list(tok_pos[0])):
                curr_pos_ids = [0 for _ in range(target_n_pos)]
                # print(sy_idx, syllable, tok_pos[1]) # TEST
                if " " == syllable:
                    pos_flag.append(curr_pos_ids)
                elif 0 == sy_idx:
                    for p_item in conv_mecab_pos:
                        tag2id = pos_tag2ids[p_item]
                        if tag2id in target_tag2ids.keys():
                            if 0 == tag2id or 1 == tag2id:
                                curr_pos_ids[0] = 1
                            else:
                                curr_pos_ids[target_tag2ids[tag2id] - 1] = 1
                    pos_flag.append(curr_pos_ids)
                else:
                    curr_pos_ids[-1] = 1
                    pos_flag.append(curr_pos_ids)
        # end loop, pos_ids

        ''' For POS Embedding '''
        max_pos_size = 10
        pos_ids = [[0 for _ in range(max_pos_size)]] # [CLS]
        for tok_pos in conv_mecab_res:
            conv_pos = [0 for _ in range(max_pos_size)]
            for p_idx, pos in enumerate(tok_pos[1]):
                if max_pos_size <= p_idx:
                    break
                filter_pos = pos if "UNKNOWN" != pos and "NA" != pos and "UNA" != pos and "VSV" != pos else "O"
                conv_pos[p_idx] = pos_tag2ids[filter_pos]
            for insert_idx in range(len(tok_pos[0])):
                if 0 == insert_idx:
                    pos_ids.append(conv_pos)
                else:
                    pos_ids.append([0 for _ in range(max_pos_size)])

        # TEST
        # for t, p in zip(text_tokens, pos_ids):
        #     print(t, p)
        # input()

        # Make NE Labels
        token_pair_len = len(text_tokens)
        label_ids = [ETRI_TAG["O"]] * token_pair_len
        b_check_use = [False for _ in range(token_pair_len)]
        for ne_idx, ne_item in enumerate(src_item.ne_list):
            # ne_char_list = list(ne_item.text.replace(" ", ""))
            ne_char_list = list(ne_item.text)
            concat_item_list = []
            for m_idx, mecab_item in enumerate(text_tokens):
                if b_check_use[m_idx]:
                    continue
                for sub_idx in range(m_idx + 1, len(text_tokens)):
                    # concat_word = ["".join(x).replace("##", "") for x in text_tokens[m_idx:sub_idx]]
                    concat_word = ["".join(x) for x in text_tokens[m_idx:sub_idx]]
                    concat_item_list.append(("".join(concat_word), (m_idx, sub_idx)))
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
        # end, Make NE Labels

        # 길이 맞춰주기
        valid_token_len = 0
        if token_max_len <= len(text_tokens):
            text_tokens = text_tokens[:token_max_len - 1]
            text_tokens.append("[SEP]")
            valid_token_len = token_max_len
        else:
            text_tokens.append("[SEP]")
            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (token_max_len - valid_token_len)

        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = ([1] * valid_token_len) + ([0] * (token_max_len - valid_token_len))
        token_type_ids = [0] * token_max_len

        ''' POS Embedding '''
        if token_max_len <= len(pos_ids):
            pos_ids = pos_ids[:token_max_len - 1]
            pos_ids.append([0 for _ in range(max_pos_size)])  # [SEP]
        else:
            pos_ids_size = len(pos_ids)
            pos_ids += [[0 for _ in range(max_pos_size)]] * (token_max_len - pos_ids_size)

        ''' POS bit flag '''
        if token_max_len <= len(pos_flag):
            pos_flag = pos_flag[:token_max_len - 1]
            pos_flag.append([0 for _ in range(target_n_pos)]) # [SEP]
        else:
            pos_flag_size = len(pos_flag)
            pos_flag += [[0 for _ in range(target_n_pos)]] * (token_max_len - pos_flag_size)

        if token_max_len <= len(label_ids):
            label_ids = label_ids[:token_max_len - 1]
            label_ids.append(ETRI_TAG["O"])
        else:
            label_ids_size = len(label_ids)
            label_ids += [ETRI_TAG["O"]] * (token_max_len - label_ids_size)

        assert len(input_ids) == token_max_len, f"{len(input_ids)}"
        assert len(attention_mask) == token_max_len, f"{len(attention_mask)}"
        assert len(token_type_ids) == token_max_len, f"{len(token_type_ids)}"
        assert len(label_ids) == token_max_len, f"{len(label_ids)}"
        assert len(pos_ids) == token_max_len, f"{len(pos_ids)}"
        assert len(pos_flag) == token_max_len, f"{len(pos_flag)}"

        # Insert npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["labels"].append(label_ids)
        npy_dict["pos_ids"].append(pos_ids)
        npy_dict["pos_flag"].append(pos_flag)

        # Debug mode
        if debug_mode:
            print(src_item.text)
            print("Wordpiece: \n", tokenizer.tokenize(src_item.text))
            print("Text tokens: \n", text_tokens)
            print("NE labels: \n", src_item.ne_list)

            # Token 별 확인
            for t_tok, lab, p in zip(text_tokens, label_ids, pos_ids):
                if "[PAD]" == t_tok:
                    break
                print(t_tok, ne_ids2tag[lab], p)
            input()

    if not debug_mode:
        save_mecab_morp_npy(npy_dict, src_list_len=total_data_count, save_dir=save_model_dir)

### MAIN ###
if "__main__" == __name__:
    print("[mecab_npy_maker] __main__ !")

    # load corpus
    pkl_src_path = "../corpus/pkl/NIKL_ne_pos.pkl"
    all_sent_list = []
    all_sent_list = load_ne_entity_list(src_path=pkl_src_path)

    # Mecab하고 모두의 말뭉치 비교
    '''
    compare_mecab_and_gold_corpus(src_corpus_list=all_sent_list)
    check_nikl_and_mecab_difference(dic_pkl_path="./mecab_cmp/mecab_compare_dict.pkl")
    check_count_morp(src_sent_list=all_sent_list)
    exit()
    mecab_pos_unk_count(all_sent_list)
    exit()
    '''

    # make *.npy (use Mecab)
    make_npy_mode = "wordpiece"
    print(f"[mecab_npy_maker] make_npy_mode: {make_npy_mode}")
    if "eojeol" == make_npy_mode:
        make_mecab_eojeol_npy(
            tokenizer_name="monologg/koelectra-base-v3-discriminator",
            src_list=all_sent_list, token_max_len=128, eojeol_max_len=50,
            debug_mode=False, josa_split=True,
            save_model_dir="mecab_split_josa_electra"
        )
    elif "wordpiece" == make_npy_mode:
        target_n_pos = 12
        make_mecab_wordpiece_npy(
            tokenizer_name="monologg/koelectra-base-v3-discriminator",
            src_list=all_sent_list, token_max_len=128, debug_mode=True,
            save_model_dir="mecab_wordpiece_electra", target_n_pos=target_n_pos
        )
    elif "morp-aware" == make_npy_mode:
        target_n_pos = 14
        target_tag_list = [  # NN은 NNG/NNP 통합
            "NNG", "NNP", "SN", "NNB", "NR", "NNBC",
            "JKS", "JKC", "JKG", "JKO", "JKB", "JX", "JC", "JKV", "JKQ",
        ]
        # target_tag_list = [v for k, v in MECAB_POS_TAG.items() if 0 != k]
        make_mecab_morp_npy(
            tokenizer_name="monologg/koelectra-base-v3-discriminator",
            src_list=all_sent_list, token_max_len=128, debug_mode=False,
            save_model_dir="mecab_morp_electra",
            target_n_pos=target_n_pos, target_tag_list=target_tag_list
        )
    elif "character" == make_npy_mode:
        target_n_pos = 15 # ##이 없기에 POS에서 첫 음절에 이어지는 부분은 맨 마지막 인덱스 사용
        target_tag_list = [  # NN은 NNG/NNP 통합
            "NNG", "NNP", "SN", "NNB", "NR", "NNBC",
            "JKS", "JKC", "JKG", "JKO", "JKB", "JX", "JC", "JKV", "JKQ",
        ]
        make_mecab_char_npy(
            tokenizer_name="monologg/kocharelectra-base-discriminator",
            src_list=all_sent_list, token_max_len=128, debug_mode=False,
            save_model_dir="mecab_char_electra",
            target_n_pos=target_n_pos, target_tag_list=target_tag_list
        )
    else:
        raise Exception(f"Check making mode: {make_npy_mode}")