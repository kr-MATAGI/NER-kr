import copy
import json
import pickle
from typing import Tuple

from data_def import Sentence, Morp, NE, Word

#==============================================================
def make_ne_mp_list(src_pair: Tuple[str, str]):
#==============================================================
    ne_file_path = src_pair[0]
    mp_file_path = src_pair[-1]

    NULL_DATA_ID_LIST = []
    with open("../corpus/pkl/NIKL_null_pos_list.pkl", mode="rb") as null_except_file:
        NULL_DATA_ID_LIST = pickle.load(null_except_file)

    # NE
    ne_json = None
    with open(ne_file_path, mode="r", encoding="utf-8") as ne_file:
        ne_json = json.load(ne_file)

    ne_sent_list = []
    doc_arr = ne_json["document"]
    for doc_obj in doc_arr:
        sent_arr = doc_obj["sentence"]
        for sent_obj in sent_arr:
            sent_data = Sentence(id=sent_obj["id"],
                                 text=sent_obj["form"])

            # null data exception
            if sent_data.id in NULL_DATA_ID_LIST:
                continue

            # NE
            ne_arr = sent_obj["NE"]
            for ne_obj in ne_arr:
                ne_data = NE(id=ne_obj["id"],
                             text=ne_obj["form"],
                             type=ne_obj["label"],
                             begin=ne_obj["begin"],
                             end=ne_obj["end"])
                sent_data.ne_list.append(copy.deepcopy(ne_data))
            ne_sent_list.append(copy.deepcopy(sent_data))
    # end, doc_arr (NE)
    print(f"NE sent list size: {len(ne_sent_list)}")

    # Morp
    mp_json = None
    with open(mp_file_path, mode="r", encoding="utf-8") as mp_file:
        mp_json = json.load(mp_file)

    morp_info_dict = {}  # sent_id: Morp
    word_info_dict = {} # sent_id: Word
    doc_arr = mp_json["document"]
    for doc_obj in doc_arr:
        sent_arr = doc_obj["sentence"]
        for sent_obj in sent_arr:
            # null data exception
            if sent_obj["id"] in NULL_DATA_ID_LIST:
                continue

            # Word
            word_arr = sent_obj["word"]
            word_data_list = []
            for word_obj in word_arr:
                word_data = Word(id=word_obj["id"],
                                 form=word_obj["form"],
                                 begin=word_obj["begin"],
                                 end=word_obj["end"])
                word_data_list.append(word_data)
            word_info_dict[sent_obj["id"]] = word_data_list

            # Morp
            mp_arr = sent_obj["morpheme"]
            mp_data_list = []
            for mp_obj in mp_arr:
                mp_data = Morp(id=mp_obj["id"],
                               form=mp_obj["form"],
                               label=mp_obj["label"],
                               word_id=mp_obj["word_id"],
                               position=mp_obj["position"])
                mp_data_list.append(mp_data)
            morp_info_dict[sent_obj["id"]] = mp_data_list
    # end, doc_arr (MP)
    print(f"Morp Info Dict Size: {len(morp_info_dict.keys())}")
    print(f"Word Info Dict Size: {len(word_info_dict.keys())}")

    # merge
    for ne_idx, ne_sent_item in enumerate(ne_sent_list):
        key_id = ne_sent_item.id
        ne_sent_list[ne_idx].morp_list = morp_info_dict[key_id]
        ne_sent_list[ne_idx].word_list = word_info_dict[key_id]

    return ne_sent_list

### MAIN ###
if "__main__" == __name__:
    nx_json_file = ("../corpus/NIKL/NXNE2102008030.json", "../corpus/NIKL/POS/NXMP1902008040.json")
    sx_json_file = ("../corpus/NIKL/SXNE2102007240.json", "../corpus/NIKL/POS/SXMP1902008031.json")

    nx_res_list = make_ne_mp_list(nx_json_file)
    print(f"nx_res_list.len : {len(nx_res_list)}")
    sx_res_list = make_ne_mp_list(sx_json_file)
    print(f"sx_res_list.len : {len(sx_res_list)}")

    merge_res_list = nx_res_list + sx_res_list
    print(f"merge_res_list.len : {len(merge_res_list)}")

    # save
    save_path = "../corpus/pkl/NIKL_ne_pos.pkl"
    with open(save_path, mode="wb") as save_file:
        pickle.dump(merge_res_list, save_file)

    # check size
    with open(save_path, mode="rb") as check_file:
        check_list = pickle.load(check_file)
        print(f"check_load.len: {len(check_list)}")