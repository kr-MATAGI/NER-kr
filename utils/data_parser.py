import json
import copy
import os
import pickle

from data_def import Sentence, Morp, NE

class ETRI_Parser:
    def __init__(self, src_dir: str):
        print(f"[ETRI_Parser][__init__] ETRI_Parser !")

        self.data_list = []
        if not os.path.exists(src_dir):
            print(f"[ETRI_Parser][__init__] ERR - NOT Existed: {src_dir}")
        else:
            self.src_dir = src_dir
            self.data_list = [x for x in os.listdir(src_dir) if ".json" in x]
            print(f"[ETRI_Parser][__init__] Total Count: {len(self.data_list)}")

    def parse_etri_json(self, src_path: str):
        ret_sent_list = []

        if not os.path.exists(src_path):
            print(f"[ETRI_Parser][__init__] ERR - Not Exist: {src_path}")
            return

        with open(src_path, mode="r", encoding="utf-8") as src_file:
            src_json = json.load(src_file)

            sent_arr = src_json["sentence"]
            for sent_obj in sent_arr:
                sent_data = Sentence(id=sent_obj["id"],
                                     text=sent_obj["text"])

                # morp
                # morp_arr = sent_obj["morp"]
                # for morp_obj in morp_arr:
                #     morp_data = Morp(id=morp_obj["id"],
                #                      lemma=morp_obj["lemma"],
                #                      type=morp_obj["type"],
                #                      position=morp_obj["position"],
                #                      weight=morp_obj["weight"])
                #     sent_data.morp_list.append(copy.deepcopy(morp_data))

                # NE
                ne_arr = sent_obj["NE"]
                for ne_obj in ne_arr:
                    ne_data = NE(id=ne_obj["id"],
                                 text=ne_obj["text"],
                                 type=ne_obj["type"],
                                 begin=ne_obj["begin"],
                                 end=ne_obj["end"],
                                 weight=ne_obj["weight"],
                                 common_noun=ne_obj["common_noun"])
                    sent_data.ne_list.append(copy.deepcopy(ne_data))
                ret_sent_list.append(copy.deepcopy(sent_data))

        return ret_sent_list

    def parse_nikl_json(self, src_path: str):
        ret_sent_list = []

        if not os.path.exists(src_path):
            print(f"[NIKL_Parser][__init__] ERR - Not Exist: {src_path}")
            return

        print(f"[NIKL_Parser][__init__] src_path: {src_path}")
        with open(src_path, mode="r", encoding="utf-8") as src_file:
            src_json = json.load(src_file)

            doc_arr = src_json["document"]
            for doc_obj in doc_arr:
                sent_arr = doc_obj["sentence"]
                for sent_obj in sent_arr:
                    sent_data = Sentence(id=sent_obj["id"],
                                         text=sent_obj["form"])

                    # NE
                    ne_arr = sent_obj["NE"]
                    for ne_obj in ne_arr:
                        ne_data = NE(id=ne_obj["id"],
                                     text=ne_obj["form"],
                                     type=ne_obj["label"],
                                     begin=ne_obj["begin"],
                                     end=ne_obj["end"])
                        sent_data.ne_list.append(copy.deepcopy(ne_data))
                    ret_sent_list.append(copy.deepcopy(sent_data))

        return ret_sent_list

### MAIN ###
if "__main__" == __name__:
    etri_parser = ETRI_Parser(src_dir="../data/old_nikl")

    # Save - ETRI 언어분석 말뭉치 (NE)
    is_save_etri = False
    if is_save_etri:
        all_sent_data = []
        for path in etri_parser.data_list:
            sent_data = etri_parser.parse_etri_json(etri_parser.src_dir+"/"+path)
            all_sent_data.extend(copy.deepcopy(sent_data))
        print(f"[data_maker][__main__] All sent data size : {len(all_sent_data)}")

        # save
        with open("../data/pkl/etri.pkl", mode="wb") as save_file:
            pickle.dump(all_sent_data, save_file)

    # Save - 모두의 말뭉치 (NE): old_version이랑 같이 쓰는데 "NE", "ne" 바꿔줘야함
    is_save_nikl = True
    if is_save_nikl:
        all_sent_data = []
        for path in etri_parser.data_list:
            sent_data = etri_parser.parse_nikl_json(etri_parser.src_dir + "/" + path)
            all_sent_data.extend(copy.deepcopy(sent_data))
        print(f"[data_maker][__main__] All sent data size : {len(all_sent_data)}")

        # save
        with open("../data/pkl/old_nikl.pkl", mode="wb") as save_file:
            pickle.dump(all_sent_data, save_file)