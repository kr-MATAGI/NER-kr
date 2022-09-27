import copy
import random
import numpy as np
import pickle

from dataclasses import dataclass, field
from collections import deque
from eunjeon import Mecab

from tag_def import ETRI_TAG, NIKL_POS_TAG, MECAB_POS_TAG
from data_def import Sentence, NE, Morp, Word

from gold_corpus_npy_maker import (
    convert_pos_tag_to_combi_tag,
    conv_NIKL_pos_giho_category,
    conv_TTA_ne_category
)

from typing import List, Dict, Tuple
from transformers import AutoTokenizer, ElectraTokenizer

## Structured
@dataclass
class Tok_Pos:
    eojeol_idx: int = -1
    tokens: List[str] = field(default=list)
    pos: List[str] = field(default=list)
    ne: str = "O"

@dataclass
class Mecab_Item:
    word: str = ""
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
def convert_mecab_pos(src_word_list: List[Word]):
#==========================================================================================
    mecab = Mecab()

    # ret_word_pair : (word, (word, pos))
    ret_word_pair_list = []
    for word_idx, word_item in enumerate(src_word_list):
        res_list = mecab.pos(word_item.form)
        new_res_list = []
        for res in res_list:
            new_res_list.append((res[0], res[1].split("+")))
        ret_word_pair_list.append((word_item.form, new_res_list))

    return ret_word_pair_list

#==========================================================================================
def tokenize_mecab_pair(mecab_pair_list, tokenizer):
#==========================================================================================
    # [ (word, [(word, [pos,...]), ...] ]
    # -> [ (word, [(tokens, [pos,...]), ...] ]

    # except_giho = ["SF", "SE", "SSO", "SSC", "SC", "SY"]
    tokenized_mecab_list = []
    for m_idx, mecab_pair in enumerate(mecab_pair_list):
        new_pos_pair_list = []
        for m_pos_idx, m_pos_pair in enumerate(mecab_pair[1]):
            tokenized_word = tokenizer.tokenize(m_pos_pair[0])
            new_pos_pair_list.append(Tok_Pos(eojeol_idx=m_idx,
                                             tokens=tokenized_word,
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
        "금리설계형의 경우 변동금리(6개월 변동 코픽스 연동형)는", "현재 중국의 공항은 400여 개다."
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

        # 매캡을 쓰면 모두의 말뭉치 morp의 word_id 정보는 사용할 수 있음
        extract_ne_list = src_item.ne_list
        # [ word, [(word, pos)] ]
        mecab_word_pair = convert_mecab_pos(src_item.word_list)
        mecab_item_list = tokenize_mecab_pair(mecab_word_pair, tokenizer)

        tok_pos_item_list = []
        for mecab_item in mecab_item_list:
            tok_pos_item_list.extend(mecab_item.tok_pos_list)

        # NE
        print(tok_pos_item_list)
        b_check_use = [False for _ in range(len(tok_pos_item_list))]
        for ne_idx, ne_item in enumerate(extract_ne_list):
            ne_char_list = list(ne_item.text.replace(" ", ""))
            concat_item_list = []
            for mec_idx, mecab_item in enumerate(tok_pos_item_list):
                if b_check_use[mec_idx]:
                    continue
                for sub_idx in range(mec_idx + 1, len(tok_pos_item_list)):
                    concat_word = ["".join(x.tokens).replace("##", "") for x in tok_pos_item_list[mec_idx:sub_idx]]
                    concat_item_list.append(("".join(concat_word), (mec_idx, sub_idx)))
            concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
            concat_item_list.sort(key=lambda x: len(x[0]))
            if 0 >= len(concat_item_list):
                continue
            target_idx_pair = concat_item_list[0][1]

            for bio_idx in range(target_idx_pair[0], target_idx_pair[1]):
                b_check_use[bio_idx] = True
                if bio_idx == target_idx_pair[0]:
                    tok_pos_item_list[bio_idx].ne = "B-" + ne_item.type
                else:
                    tok_pos_item_list[bio_idx].ne = "I-" + ne_item.type

        # PRINT
        print(tok_pos_item_list)

        text_tokens = []
        pos_ids = []
        for tp_idx, tok_pos in enumerate(tok_pos_item_list):
            tok_pos
            if
        print(text_tokens)

        input()

### MAIN ###
if "__main__" == __name__:
    print("[mecab_npy_maker] __main__ !")

    # load corpus
    pkl_src_path = "../corpus/pkl/NIKL_ne_pos.pkl"
    all_sent_list = load_ne_entity_list(src_path=pkl_src_path)

    # make *.npy (use Mecab)
    make_mecab_eojeol_npy(
        tokenizer_name="monologg/koelectra-base-v3-discriminator",
        src_list=all_sent_list, token_max_len=128, eojeol_max_len=50,
        debug_mode=False, josa_split=False,
        save_model_dir="mecab_eojeol_electra"
    )