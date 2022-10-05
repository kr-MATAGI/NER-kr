import numpy as np
import random
import pickle
from kiwipiepy import Kiwi
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import deque

from tag_def import ETRI_TAG, NIKL_POS_TAG, MECAB_POS_TAG
from data_def import Sentence, NE, Morp, Word

## Global
random.seed(42)
np.random.seed(42)

## Dataclass
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
def load_ne_entity_list(src_path: str = ""):
#==========================================================================================
    all_sent_list = []

    with open(src_path, mode="rb") as pkl_file:
        all_sent_list = pickle.load(pkl_file)
        print(f"[mecab_npy_maker][load_ne_entity_list] all_sent_list size: {len(all_sent_list)}")
    # all_sent_list = conv_TTA_ne_category(all_sent_list)

    return all_sent_list

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
def conv_kiwi_pos_to_nikl(kiwi_pos):
#==========================================================================================
    ret_pos = kiwi_pos.replace("-R", "").replace("-I", "")
    return ret_pos

#==========================================================================================
def compare_kiwipiepy_and_gold_corpus(src_corpus_list: list):
#==========================================================================================
    '''
        cmp_results_dict = { key: POS, value: { [(sent_id, sent_text, [(word_id, gold_corpus, mecab_pos_concat]] }
    '''
    cmp_results_dict = {}

    # Kiwi
    kiwi = Kiwi()
    for sent_idx, sent_item in enumerate(src_corpus_list):
        if 0 == (sent_idx % 100):
            print(f"{sent_idx} is processing...")
        # Gold Corpus
        gold_corpus_dict = {}  # key: word_id, value = Word_POS_pair(word, pos)

        # Set word_id(key)
        for word_item in sent_item.word_list:
            gold_corpus_dict[word_item.id] = []

        if sent_item.text != "필리핀 국민의 약 10%인 800만 명은 세계 곳곳에서 건설노동자 가정부 유모 등으로 힘들게 일한다.":
            continue

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
        kiwi_res_dict = {} # key: word_id, value = Word_POS_pair(word, pos)
        kiwi_res_deque = deque()
        kiwi.analyze()
        for kiwi_res in kiwi.tokenize(sent_item.text):
            kiwi_res_deque.append([kiwi_res.form, kiwi_res.tag, kiwi_res.start, kiwi_res.len])

        pos_ids = 0
        eojeol_idx = 1
        target_list = []
        prev_start_idx = -1
        while 0 != len(kiwi_res_deque):
            pop_item = kiwi_res_deque.popleft()
            if pos_ids != pop_item[2] and prev_start_idx != pop_item[2]:
                pos_ids = pop_item[2]
                concat_morp = [x[0] for x in target_list]
                concat_pos = []
                for x in target_list:
                    conv_pos = conv_kiwi_pos_to_nikl(x[1])
                    concat_pos.append(conv_pos)
                word_pos_pair = Morp_pair(morp=concat_morp, pos=concat_pos)
                kiwi_res_dict[eojeol_idx] = word_pos_pair
                eojeol_idx += 1
                target_list.clear()

            target_list.append([pop_item[0], pop_item[1]])
            if prev_start_idx != pop_item[2]:
                pos_ids += pop_item[-1]
            prev_start_idx = pop_item[2]

        if 0 < len(target_list):
            concat_morp = [x[0] for x in target_list]
            concat_pos = []
            for x in target_list:
                conv_pos = conv_kiwi_pos_to_nikl(x[1])
                concat_pos.append(conv_pos)
            word_pos_pair = Morp_pair(morp=concat_morp, pos=concat_pos)
            kiwi_res_dict[eojeol_idx] = word_pos_pair

        # Compare
        ignore_list = ["SF", "SE", "SS", "SP", "SO", "SW", "SSO", "SSC", "SC", "SY"] # "SL", "SH", "SN"
        for key, value in gold_corpus_dict.items():
            conv_nikl_giho_value = ["GH" if x in ignore_list else x for x in value.pos]

            kiwi_dict_value = kiwi_res_dict[key]
            conv_kiwi_giho_value = ["GH" if x in ignore_list else x for x in kiwi_dict_value.pos]
            if "+".join(conv_nikl_giho_value) != "+".join(conv_kiwi_giho_value):
                # 형태소 분석 정보의 길이가 같다
                if len(conv_nikl_giho_value) == len(conv_kiwi_giho_value):
                    for idx, (nikl_pos, mecab_pos) in enumerate(zip(conv_nikl_giho_value, conv_kiwi_giho_value)):
                        if nikl_pos != mecab_pos:
                            filtered_target = (idx, (value.morp[idx], value.pos[idx]))
                            cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                                  target=filtered_target,
                                                  nikl_morp=value, mecab_morp=kiwi_dict_value)
                            if nikl_pos not in cmp_results_dict.keys():
                                cmp_results_dict[nikl_pos] = [cmp_res]
                            else:
                                cmp_results_dict[nikl_pos].append(cmp_res)
                # gold corpus의 형태소 분석 정보가 더 많다.
                elif len(conv_nikl_giho_value) > len(conv_kiwi_giho_value):
                    filtered_idx = -1
                    for idx, mecab_pos in enumerate(conv_kiwi_giho_value):
                        if mecab_pos != conv_nikl_giho_value[idx]:
                            filtered_idx = idx
                            break
                    filtered_target = ()
                    cmp_res = Compare_res()
                    if -1 != filtered_idx:
                        filtered_target = (filtered_idx, (value.morp[filtered_idx], value.pos[filtered_idx]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target,
                                              nikl_morp=value, mecab_morp=kiwi_dict_value)
                    else:
                        filtered_target = (0, (value.morp[0], value.pos[0]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target, nikl_morp=value, mecab_morp=kiwi_dict_value)
                    if value.pos[filtered_idx] not in cmp_results_dict.keys():
                        cmp_results_dict[value.pos[filtered_idx]] = [cmp_res]
                    else:
                        cmp_results_dict[value.pos[filtered_idx]].append(cmp_res)
                # kiwi의 형태소 분석 정보가 더 많다.
                else:
                    filtered_idx = -1
                    for idx, nikl_pos in enumerate(conv_nikl_giho_value):
                        if nikl_pos != conv_kiwi_giho_value[idx]:
                            filtered_idx = idx
                            break
                    filtered_target = ()
                    cmp_res = Compare_res()
                    if -1 != filtered_idx:
                        filtered_target = (filtered_idx, (value.morp[filtered_idx], value.pos[filtered_idx]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target,
                                              nikl_morp=value, mecab_morp=kiwi_dict_value)
                    else:
                        filtered_target = (0, (value.morp[0], value.pos[0]))
                        cmp_res = Compare_res(sent_id=sent_item.id, sent_text=sent_item.text,
                                              target=filtered_target,
                                              nikl_morp=value, mecab_morp=kiwi_dict_value)
                    if value.pos[filtered_idx] not in cmp_results_dict.keys():
                        cmp_results_dict[value.pos[filtered_idx]] = [cmp_res]
                    else:
                        cmp_results_dict[value.pos[filtered_idx]].append(cmp_res)

    # Complete compare
    for key, value in cmp_results_dict.items():
        with open("./kiwi_cmp/"+key+".txt", mode="w", encoding="utf-8") as write_file:
            print(f"Value Size: {len(value)}")
            for v in value:
                write_file.write("sent_id: "+v.sent_id+"\n")
                write_file.write("sent_text: "+v.sent_text+"\n")
                write_file.write("target: "+str(v.target)+"\n")
                write_file.write("nikl_item: "+str(v.nikl_morp)+"\n")
                write_file.write("mecab_item: " + str(v.mecab_morp) + "\n")
                write_file.write("\n\n")

    # Save Dictionary
    with open("./mecab_cmp/kiwi_compare_dict.pkl", mode="wb") as write_pkl:
        pickle.dump(cmp_results_dict, write_pkl)
        print("save pkl")

### MAIN ###
if "__main__" == __name__:
    print("[kiwipiepy_utils] __main__ !")

    # load corpus
    pkl_src_path = "../corpus/pkl/NIKL_ne_pos.pkl"
    all_sent_list = []
    all_sent_list = load_ne_entity_list(src_path=pkl_src_path)
    compare_kiwipiepy_and_gold_corpus(all_sent_list)