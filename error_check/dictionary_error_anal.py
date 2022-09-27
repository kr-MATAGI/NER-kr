import os
import numpy as np
import torch
import pickle
import pandas as pd

from collections import deque, defaultdict
from typing import List

import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/NER_Private")
from transformers import ElectraTokenizer
from model.electra_eojeol_model import Electra_Eojeol_Model
from model.electra_lstm_crf import ELECTRA_POS_LSTM
from utils.tag_def import ETRI_TAG, NIKL_POS_TAG

from seqeval.metrics.sequence_labeling import get_entities

#===========================================================================
def load_dataset_by_path(datasets_path: str = ""):
#===========================================================================
    ret_dict = {
        "dev_npy": None,
        "labels": None,
        "pos_tag": None,
        "eojeol_ids": None
    }

    # load data
    ret_dict["dev_npy"] = np.load(datasets_path+"/dev.npy")
    ret_dict["labels"] = np.load(datasets_path+"/dev_labels.npy")
    ret_dict["pos_tag"] = np.load(datasets_path+"/dev_pos_tag.npy")
    ret_dict["eojeol_ids"] = np.load(datasets_path+"/dev_eojeol_ids.npy")
    print(f"dev.shape - npy: {ret_dict['dev_npy'].shape}, label: {ret_dict['labels'].shape}, " 
          f"pos_tag: {ret_dict['pos_tag'].shape}, eojeol_ids: {ret_dict['eojeol_ids'].shape}")

    return ret_dict

#===========================================================================
def make_error_dictionary(
        model_path: str = "", datasets_path: str = "",
        model_name: str = "", save_dir_path: str = ""
):
#===========================================================================
    error_dict = {}

    # Load model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = Electra_Eojeol_Model.from_pretrained(model_path)

    # Load Datasets
    datasets_dict = load_dataset_by_path(datasets_path=datasets_path)

    model.eval()
    total_data_size = datasets_dict["dev_npy"].shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}

    for data_idx in range(total_data_size):
        if 0 == (data_idx % 100):
            print(f"{data_idx} is processing...")

        inputs = {
            "input_ids": torch.LongTensor([datasets_dict["dev_npy"][data_idx, :, 0]]),
            "attention_mask": torch.LongTensor([datasets_dict["dev_npy"][data_idx, :, 1]]),
            "token_type_ids": torch.LongTensor([datasets_dict["dev_npy"][data_idx, :, 2]]),
            "labels": torch.LongTensor([datasets_dict["labels"][data_idx, :]]),
            "eojeol_ids": torch.LongTensor([datasets_dict["eojeol_ids"][data_idx, :]]),
            "pos_tag_ids": torch.LongTensor([datasets_dict["pos_tag"][data_idx, :]])
        }

        # Make Eojeol
        text: List[str] = []
        eojeol_ids = datasets_dict["eojeol_ids"][data_idx, :]
        merge_idx = 0
        for eojeol_cnt in eojeol_ids:
            if 0 >= eojeol_cnt:
                break
            eojeol_tokens = datasets_dict["dev_npy"][data_idx, :, 0][merge_idx:merge_idx + eojeol_cnt]
            merge_idx += eojeol_cnt
            conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
            text.append(conv_eojeol_tokens)

        # Model
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()

        preds = np.array(logits)[0]
        preds = np.argmax(preds, axis=1)
        labels = datasets_dict["labels"][data_idx, :]

        pos_ids = datasets_dict["pos_tag"][data_idx, :]
        columns = ["text", "labels", "preds", "pos"]
        row_list = []
        for p_idx in range(len(text)):
            conv_label = ne_ids_to_tag[labels[p_idx]]
            conv_preds = ne_ids_to_tag[preds[p_idx]]
            conv_pos = [pos_ids_to_tag[x] for x in pos_ids[p_idx]]
            row_list.append([text[p_idx], conv_label, conv_preds, conv_pos])
        pd_df = pd.DataFrame(row_list, columns=columns)

        for df_idx, df_item in pd_df.iterrows():
            if 0 >= len(df_item["text"]):
                break
            if df_item["labels"] != df_item["preds"]:
                merge_pos = df_item["pos"]
                merge_pos = [x for x in merge_pos if "O" != x]
                merge_pos = "+".join(merge_pos)
                if merge_pos not in error_dict.keys():
                    error_dict[merge_pos] = [(data_idx, df_item["text"], df_item["labels"], df_item["preds"])]
                else:
                    error_dict[merge_pos].append((data_idx, df_item["text"], df_item["labels"], df_item["preds"]))

    # Write
    for key in error_dict.keys():
        with open(save_dir_path+"/"+key+".txt", mode="w") as write_file:
            err_info_list = error_dict[key]
            for err_info in err_info_list:
                write_file.write(str(err_info[0])+"\t"+err_info[1]+"\t"+err_info[2]+"\t"+err_info[3]+"\n")
    # Sorted
    err_items_sorted = sorted(error_dict.items(), key=lambda x: len(x[1]), reverse=True)
    for err_item in err_items_sorted:
        print(err_item[0], err_item[1])
        print("=================================================================\n")

#===========================================================================
def search_outputs_by_data_idx(
        model_path: str = "", datasets_path: str = "",
        model_name: str = ""
):
#===========================================================================
    # Load model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = Electra_Eojeol_Model.from_pretrained(model_path)

    # Load Datasets
    datasets_dict = load_dataset_by_path(datasets_path=datasets_path)

    model.eval()
    total_data_size = datasets_dict["dev_npy"].shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}

    while True:
        print(">>> input target_idx:")
        target_idx = int(input())
        for data_idx in range(total_data_size):
            if target_idx != data_idx:
                continue

            inputs = {
                "input_ids": torch.LongTensor([datasets_dict["dev_npy"][data_idx, :, 0]]),
                "attention_mask": torch.LongTensor([datasets_dict["dev_npy"][data_idx, :, 1]]),
                "token_type_ids": torch.LongTensor([datasets_dict["dev_npy"][data_idx, :, 2]]),
                "labels": torch.LongTensor([datasets_dict["labels"][data_idx, :]]),
                "eojeol_ids": torch.LongTensor([datasets_dict["eojeol_ids"][data_idx, :]]),
                "pos_tag_ids": torch.LongTensor([datasets_dict["pos_tag"][data_idx, :]])
            }

            # Make Eojeol
            text: List[str] = []
            eojeol_ids = datasets_dict["eojeol_ids"][data_idx, :]
            merge_idx = 0
            for eojeol_cnt in eojeol_ids:
                if 0 >= eojeol_cnt:
                    break
                eojeol_tokens = datasets_dict["dev_npy"][data_idx, :, 0][merge_idx:merge_idx + eojeol_cnt]
                merge_idx += eojeol_cnt
                conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
                text.append(conv_eojeol_tokens)

            # Model
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()

            preds = np.array(logits)[0]
            preds = np.argmax(preds, axis=1)
            labels = datasets_dict["labels"][data_idx, :]

            pos_ids = datasets_dict["pos_tag"][data_idx, :]
            columns = ["text", "labels", "preds", "pos"]
            row_list = []
            for p_idx in range(len(text)):
                conv_label = ne_ids_to_tag[labels[p_idx]]
                conv_preds = ne_ids_to_tag[preds[p_idx]]
                conv_pos = [pos_ids_to_tag[x] for x in pos_ids[p_idx]]
                row_list.append([text[p_idx], conv_label, conv_preds, conv_pos])
            pd_df = pd.DataFrame(row_list, columns=columns)
            print(pd_df)

#===========================================================================
def ranking_by_read_file(root_dir: str = ""):
#===========================================================================
    target_files = os.listdir(root_dir)
    res_dict = {}
    for file in target_files:
        file_path = root_dir + "/" + file
        key = file.replace(".txt", "")

        print(file_path)
        with open(file_path, mode="r") as read_file:
            res_dict[key] = len(read_file.readlines())
    print(sorted(res_dict.items(), key=lambda x: x[1], reverse=True))

#===========================================================================
def divide_by_category(root_dir: str = ""):
#===========================================================================
    target_files = os.listdir(root_dir)
    res_dict = {}
    for file in target_files:
        file_path = root_dir + "/" + file
        pos = file.replace(".txt", "")

        print(file_path)
        with open(file_path, mode="r") as read_file:
            for r_line in  read_file.readlines():
                item = r_line.replace("\n", "").split("\t")
                if item[2] not in res_dict.keys():
                    res_dict[item[2]] = [(item[0], item[1], item[-1], pos)]
                else:
                    res_dict[item[2]].append((item[0], item[1], item[-1], pos))

    # Write
    for key in res_dict.keys():
        with open("./dir_err_results_by_ne" + "/" + key + ".txt", mode="w") as write_file:
            err_info_list = res_dict[key]
            for err_info in err_info_list:
                write_file.write(
                    str(err_info[0]) + "\t" + err_info[1] + "\t" + err_info[2] + "\t" + err_info[3] + "\n")

def compare(f)

### MAIN ###
if "__main__" == __name__:
    # model_path = "./old_eojeol_model/model"
    # datasets_path = "./old_eojeol_model/npy"
    model_path = "./eojeol_model/model"
    datasets_path = "./eojeol_model/npy"
    model_name = "monologg/koelectra-base-v3-discriminator"
    save_dir_path = "./dic_err_results"
    # make_error_dictionary(model_path=model_path, model_name=model_name,
    #                       datasets_path=datasets_path, save_dir_path=save_dir_path)

    # ranking_by_read_file(save_dir_path)
    #
    # divide_by_category(root_dir=save_dir_path)
    #
    search_outputs_by_data_idx(model_path=model_path, model_name=model_name,
                               datasets_path=datasets_path)


