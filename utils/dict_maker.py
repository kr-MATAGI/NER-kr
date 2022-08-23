import copy
import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET

from typing import List, Dict
from dataclasses import dataclass, field

from transformers import AutoTokenizer

### Data def
@dataclass
class Word_Info:
    word: str = ""
    word_unit: str = ""
    word_type: str = ""

@dataclass
class Sense_Info:
    sense_no: str = ""
    pos: str = ""
    type: str = ""
    definition: str = ""

@dataclass
class Dict_Item:
    target_code: int = field(init=False, default=0)
    word_info: Word_Info = field(init=False, default=Word_Info())
    sense_info: Sense_Info = field(init=False, default=Sense_Info())

#===============================================================
def read_korean_dict_xml(src_dir_path: str):
#===============================================================
    ret_kr_dict_item_list: List[Dict_Item] = []

    print(f"[dict_utils][read_korean_dict_xml] src_dir_path: {src_dir_path}")
    src_dict_list = [src_dir_path+"/"+path for path in os.listdir(src_dir_path)]
    print(f"[dict_utils][read_korean_dict_xml] size: {len(src_dict_list)},\n{src_dict_list}")

    '''
        @NOTE
            989450_1100000.xml 파일의 1474638~9 line에 parse 되지 않는 토큰 있음
            989450_550000.xml 파일의 1035958~9 line에 parse 되지 않는 토큰 있음
    '''
    for src_idx, src_dict_xml in enumerate(src_dict_list):
        print(f"[dict_utils][read_korean_dict_xml] {src_idx+1}, {src_dict_xml}")
        document = ET.parse(src_dict_xml)
        root = document.getroot() # <channel>

        for item_tag in root.iter(tag="item"):
            dict_item = Dict_Item()
            dict_item.target_code = int(item_tag.find("target_code").text)

            # word info
            word_info = Word_Info()
            word_info_tag = item_tag.find("wordInfo")
            word_info.word = word_info_tag.find("word").text.replace("^", "").replace("-", "")
            word_info.word_unit = word_info_tag.find("word_unit").text
            word_info.word_type = word_info_tag.find("word_type").text

            # sense info
            sense_info = Sense_Info()
            sense_info_tag = item_tag.find("senseInfo")
            sense_info.sense_no = sense_info_tag.find("sense_no").text
            if None != sense_info_tag.find("pos"):
                sense_info.pos = sense_info_tag.find("pos").text
            sense_info.type = sense_info_tag.find("type").text
            sense_info.definition = sense_info_tag.find("definition").text

            dict_item.word_info = copy.deepcopy(word_info)
            dict_item.sense_info = copy.deepcopy(sense_info)

            ret_kr_dict_item_list.append(dict_item)

    print(f"[dict_utils][read_korean_dict_xml] total dict item size: {len(ret_kr_dict_item_list)}")
    return ret_kr_dict_item_list

#===============================================================
def make_dict_hash_table(dict_path: str):
#===============================================================
    dict_data_list: List[Dict_Item] = []

    # load
    with open(dict_path, mode="rb") as dict_pkl:
        dict_data_list = pickle.load(dict_pkl)
        print(f"[dict_utils][make_dict_hash_table] dic_data_list.size: {len(dict_data_list)}")

    # dict
    hash_dict: Dict[str, List[Dict_Item]] = {}
    for dic_idx, dic_data in enumerate(dict_data_list):
        if 0 == (dic_idx % 10000):
            print(f"[dict_utils][make_dict_hash_table] {dic_idx} is processing... {dic_data.word_info.word}")

        first_word = dic_data.word_info.word[0]
        if first_word in hash_dict.keys():
            hash_dict[first_word].append(dic_data)
        else:
            hash_dict[first_word] = []

    return hash_dict

#===============================================================
def make_dict_boundary_npy(corpus_npy_path: str, dict_hash: Dict[str, Dict_Item], tokenizer_name: str,
                           mode="train"):
#===============================================================
    # init
    concat_limit = 5
    max_seq_len = 128
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # load
    target_npy_file = os.listdir(corpus_npy_path)
    target_npy_file = [x for x in target_npy_file if mode+".npy" == x][0]

    npy_datasets = np.load(corpus_npy_path+"/"+target_npy_file)
    print(f"[dict_utils][make_dict_boundary_npy] mode: {mode}, npy.shape: {npy_datasets.shape}")

    for npy_idx, npy_data in enumerate(npy_datasets):
        input_ids = npy_data[:, 0]
        end_ids = np.where(input_ids == 3)[0][0]

        decode_sent = tokenizer.decode(input_ids)
        print(input_ids, end_ids, decode_sent)

        ne_ids = 1
        for t_idx in range(1, end_ids):
            is_matching = False
            concat_word = ""
            for concat_cnt in range(concat_limit):
                if end_ids <= (t_idx + concat_cnt):
                    break
                add_word = tokenizer.convert_ids_to_tokens([input_ids[t_idx + concat_cnt]])[0]
                if "##" in add_word:
                    concat_word += add_word.replace("##", "")
                else:
                    if 0 >= len(concat_word) or (add_word in [".", "(", ")", "[", "]", "{", "}", "<", ">"]):
                        concat_word += add_word
                    else:
                        concat_word += " " + add_word

                # search in dict
                if concat_word[0] in dict_hash.keys():
                    dict_list: List[Dict_Item] = []
                    for src_dict_item in dict_hash[concat_word[0]]:
                        conv_dict_item = copy.deepcopy(src_dict_item)
                        conv_dict_item.word_info.word = src_dict_item.word_info.word.replace("^", "").replace("-", "")
                        dict_list.append(conv_dict_item)
                    for dict_item in dict_list:
                        if concat_word.replace(" ", "") == dict_item.word_info.word:
                            is_matching = True
                            print(concat_word, "\n", dict_item.word_info, "\n", dict_item.sense_info)
                            input()

    exit()

### Main
if "__main__" == __name__:
    is_save_pkl = True
    if is_save_pkl:
        dir_path = "../data/dict/우리말샘"
        res_kr_dict_item_list = read_korean_dict_xml(src_dir_path=dir_path)

        # save *.pkl
        save_path = "../우리말샘_dict.pkl"
        with open(save_path, mode="wb") as save_pkl:
            pickle.dump(res_kr_dict_item_list, save_pkl)
            print(f"[dict_utils][__main__] Complete save - {save_path}")

    is_make_npy = False
    if is_make_npy:
        res_hash_dict = make_dict_hash_table(dic_path="../우리말샘_dict.pkl")
        make_dict_boundary_npy(corpus_npy_path="../data/npy/old_nikl/bert", dict_hash=res_hash_dict,
                               tokenizer_name="klue/bert-base", mode="train")
        make_dict_boundary_npy(corpus_npy_path="../data/npy/old_nikl/bert", dict_hash=res_hash_dict,
                               tokenizer_name="klue/bert-base", mode="dev")
        make_dict_boundary_npy(corpus_npy_path="../data/npy/old_nikl/bert", dict_hash=res_hash_dict,
                               tokenizer_name="klue/bert-base", mode="test")