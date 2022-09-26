import copy
import random
import numpy as np
import pickle

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
        # "token_seq_len": [],
        "pos_tag_ids": [],
        "eojeol_ids": [],
        # "entity_ids": [], # for token_type_ids (bert segment embedding)
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

        # make (word, token, pos) pair
        # [(word, [tokens], [POS])]
        word_tokens_pos_pair_list: List[Tuple[str, List[str], Tuple[str, str]]] = []
        split_giho_label = [
            "SF", "SE", "SSO", "SSC", "SC", "SY"
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
        if josa_split:
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

        # POS

        # Eojoel boundary
        eojeol_boundary_list: List[int] = []
        for wtp_ids, wtp_item in enumerate(word_tokens_pos_pair_list):
            token_size = len(wtp_item[1])
            eojeol_boundary_list.append(token_size)

        # Sequence Length
        # 토큰 단위
        valid_token_len = 0
        text_tokens.insert(0, "[CLS]")
        text_tokens.append("[SEP]")
        if token_max_len <= len(text_tokens):
            text_tokens = text_tokens[:token_max_len - 1]
            text_tokens.append("[SEP]")
            valid_token_len = token_max_len
        else:
            valid_token_len = len(text_tokens)
            text_tokens += ["[PAD]"] * (token_max_len - valid_token_len)

        attention_mask = ([1] * valid_token_len) + ([0] * (token_max_len - valid_token_len))
        token_type_ids = [0] * token_max_len
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        # 어절 단위
        # label_ids
        valid_eojeol_len = 0
        labels_ids.insert(0, ETRI_TAG["O"])
        if eojeol_max_len <= len(labels_ids):
            labels_ids = labels_ids[:eojeol_max_len - 1]
            labels_ids.append(ETRI_TAG["O"])
            valid_eojeol_len = eojeol_max_len
        else:
            labels_ids_size = len(labels_ids)
            valid_eojeol_len = labels_ids_size
            for _ in range(eojeol_max_len - labels_ids_size):
                labels_ids.append(ETRI_TAG["O"])

        # pos_tag_ids
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

        assert len(input_ids) == token_max_len, f"{len(input_ids)} + {input_ids}"
        assert len(attention_mask) == token_max_len, f"{len(attention_mask)} + {attention_mask}"
        assert len(token_type_ids) == token_max_len, f"{len(token_type_ids)} + {token_type_ids}"
        assert len(labels_ids) == eojeol_max_len, f"{len(labels_ids)} + {labels_ids}"
        assert len(pos_tag_ids) == eojeol_max_len, f"{len(pos_tag_ids)} + {pos_tag_ids}"
        assert len(eojeol_boundary_list) == eojeol_max_len, f"{len(eojeol_boundary_list)} + {eojeol_boundary_list}"

        # add to npy_dict
        npy_dict["input_ids"].append(input_ids)
        npy_dict["attention_mask"].append(attention_mask)
        npy_dict["token_type_ids"].append(token_type_ids)
        npy_dict["labels"].append(labels_ids)

        # convert tags
        pos_tag_ids = convert_pos_tag_to_combi_tag(pos_tag_ids)
        npy_dict["pos_tag_ids"].append(pos_tag_ids)
        npy_dict["eojeol_ids"].append(eojeol_boundary_list)

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
            for wtpp, la, pti, ej_b in zip(temp_word_tokens_pos_pair_list, labels_ids, debug_pos_tag_ids,
                                               eojeol_boundary_list):
                # pos_array = np.array(pti)
                # if (4 < np.where(pos_array != 'O')[0].size) and (2 <= np.where(pos_array == 'NNP')[0].size):
                print(wtpp[0], ne_ids2tag[la], pti, wtpp[1], ej_b)
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