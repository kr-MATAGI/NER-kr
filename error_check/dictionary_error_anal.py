import os
import numpy as np
import torch
import pickle
import pandas as pd

from eunjeon import Mecab
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
        with open(save_dir_path+"/"+key+".txt", mode="w", encoding="utf-8") as write_file:
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
    total_errors = 0
    target_josa = ["JX", "JC", "JKS", "JKC", "JKG", "JKO", "JKB"] # XSN, VCP
    target_nn = ["NNG", "NNP", "CONCAT_NN"]
    nn_josa_errors = 0
    for file in target_files:
        file_path = root_dir + "/" + file
        key = file.replace(".txt", "")

        print(file_path)
        with open(file_path, mode="r", encoding="utf-8") as read_file:
            res_dict[key] = len(read_file.readlines())
            total_errors += res_dict[key]
            nn_cnt = [True for x in target_nn if x in key]
            josa_cnt = [True for x in target_josa if x in key]
            if 0 < len(nn_cnt) and 0 < len(josa_cnt):
                nn_josa_errors += res_dict[key]
    print(sorted(res_dict.items(), key=lambda x: x[1], reverse=True))
    print(f"total_errors: {total_errors}, nn_josa_errors: {nn_josa_errors}")

#===========================================================================
def divide_by_category(root_dir: str = ""):
#===========================================================================
    target_files = os.listdir(root_dir)
    res_dict = {}
    for file in target_files:
        file_path = root_dir + "/" + file
        pos = file.replace(".txt", "")

        print(file_path)
        with open(file_path, mode="r", encoding="utf-8") as read_file:
            for r_line in read_file.readlines():
                item = r_line.replace("\n", "").split("\t")
                if item[2] not in res_dict.keys():
                    res_dict[item[2]] = [(item[0], item[1], item[-1], pos)]
                else:
                    res_dict[item[2]].append((item[0], item[1], item[-1], pos))

    # Write
    save_dir_path = "./dir_err_results_by_ne"
    for key in res_dict.keys():
        with open(save_dir_path + "/" + key + ".txt", mode="w", encoding="utf-8") as write_file:
            err_info_list = res_dict[key]
            for err_info in err_info_list:
                write_file.write(
                    str(err_info[0]) + "\t" + err_info[1] + "\t" + err_info[2] + "\t" + err_info[3] + "\n")

#===========================================================================
def compare_josa_split_results(
        model_path: str = "", datasets_path: str = "",
        model_name: str = "", error_dir_path: str = ""
):
#===========================================================================
    # Load Model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = Electra_Eojeol_Model.from_pretrained(model_path)
    mecab = Mecab()

    # Load Datasets
    datasets_dict = load_dataset_by_path(datasets_path=datasets_path)

    model.eval()
    total_data_size = datasets_dict["dev_npy"].shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}

    # Count Info
    total_err_count = 0
    josa_err_count = 0
    fixed_error_count = 0
    fixed_ne_dict = {}

    target_josa = ["JX", "JC", "JKS", "JKC", "JKG", "JKO", "JKB"] # XSN, VCP
    target_files = os.listdir(error_dir_path)
    for file_name in target_files:
        file_path = error_dir_path + "/" + file_name
        bio_tag = file_name.replace(".txt", "")

        with open(file_path, mode="r", encoding="utf-8") as read_file:
            for read_line in read_file.readlines():
                total_err_count += 1

                line_item = read_line.replace("\n", "").split("\t")
                target_idx = int(line_item[0])

                # For Debug by idx
                # if target_idx != 1212:
                #     continue

                target_text = line_item[1]
                mecab_text = mecab.pos(target_text)
                mecab_text = [x[0] for x in mecab_text if x[1] not in target_josa]
                target_text = "".join(mecab_text)
                target_preds = line_item[2]
                target_pos = line_item[3]

                is_target = [x for x in target_josa if x in target_pos]
                if 0 < len(is_target):
                    josa_err_count += 1
                    is_target_str = "+".join(is_target)
                    if is_target_str not in fixed_ne_dict.keys():
                        fixed_ne_dict[is_target_str] = []

                    print(line_item, "target: ", target_text)
                    inputs = {
                        "input_ids": torch.LongTensor([datasets_dict["dev_npy"][target_idx, :, 0]]),
                        "attention_mask": torch.LongTensor([datasets_dict["dev_npy"][target_idx, :, 1]]),
                        "token_type_ids": torch.LongTensor([datasets_dict["dev_npy"][target_idx, :, 2]]),
                        "labels": torch.LongTensor([datasets_dict["labels"][target_idx, :]]),
                        "eojeol_ids": torch.LongTensor([datasets_dict["eojeol_ids"][target_idx, :]]),
                        "pos_tag_ids": torch.LongTensor([datasets_dict["pos_tag"][target_idx, :]])
                    }
                    # Model
                    outputs = model(**inputs)
                    logits = outputs.logits.detach().cpu().numpy()

                    # Make Eojeol
                    text: List[str] = []
                    eojeol_ids = datasets_dict["eojeol_ids"][target_idx, :]
                    merge_idx = 0
                    for eojeol_cnt in eojeol_ids:
                        if 0 >= eojeol_cnt:
                            break
                        eojeol_tokens = datasets_dict["dev_npy"][target_idx, :, 0][merge_idx:merge_idx + eojeol_cnt]
                        merge_idx += eojeol_cnt
                        conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
                        text.append(conv_eojeol_tokens)

                    preds = np.array(logits)[0]
                    preds = np.argmax(preds, axis=1)
                    labels = datasets_dict["labels"][target_idx, :]
                    # pos_ids = datasets_dict["pos_tag"][target_idx, :]
                    candi_list = []
                    for p_idx in range(len(text)):
                        conv_label = ne_ids_to_tag[labels[p_idx]]
                        conv_preds = ne_ids_to_tag[preds[p_idx]]
                        # conv_pos = [pos_ids_to_tag[x] for x in pos_ids[p_idx]]
                        if text[p_idx] in target_text:
                            candi_list.append((text[p_idx], conv_label, conv_preds))
                    # candi_list = [x for x in candi_list if x[1] == bio_tag]
                    if 0 >= len(candi_list):
                        continue
                    candi_list = sorted(candi_list, key=lambda x: len(x[0]), reverse=True)
                    print(candi_list)
                    # input()
                    if candi_list[0][1] == candi_list[0][2] and candi_list[0][1] != target_preds: #and (candi_list[0][1] == bio_tag)
                        fixed_error_count += 1
                        # candi_list[0][0] : text, candi_list[0][1] : label, candi_list[0][2] : preds
                        fixed_ne_dict[is_target_str].append((target_idx, candi_list[0][0], candi_list[0][1],
                                                             candi_list[0][2], line_item[1], target_preds))
    print(fixed_ne_dict)
    print(total_err_count)
    print(josa_err_count)
    print(fixed_error_count)

    # Save fixed_ne_dict
    with open("./fixed_ne_dict.pkl", "wb") as write_pkl:
        pickle.dump(fixed_ne_dict, write_pkl)


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
    # exit()

    save_dir_path = "./dic_err_results"
    # divide_by_category(root_dir=save_dir_path)
    #
    ne_err_results_save_path = "./dir_err_results_by_ne"
    search_outputs_by_data_idx(model_path=model_path, model_name=model_name,
                               datasets_path=datasets_path)

    # compare_josa_split_results(model_path=model_path, model_name=model_name,
    #                            datasets_path=datasets_path,
    #                            error_dir_path=ne_err_results_save_path)
