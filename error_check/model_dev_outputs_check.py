import numpy as np
import torch
import pickle
import pandas as pd

from collections import deque, defaultdict
from typing import List

import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/NER_Private")
from transformers import ElectraModel, ElectraTokenizer
from model.electra_eojeol_model import Electra_Eojeol_Model
from model.electra_lstm_crf import ELECTRA_POS_LSTM
from utils.tag_def import ETRI_TAG, NIKL_POS_TAG

from seqeval.metrics.sequence_labeling import get_entities

#===========================================================================
def predict_eojeol_validation_set(
        model_path: str = "", datasets_path: str = "",
        model_name: str = "", search_mode: bool = False
):
#===========================================================================
    # load model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = Electra_Eojeol_Model.from_pretrained(model_path)

    # load data
    dev_npy_np = np.load(datasets_path+"/dev.npy")
    dev_label_np = np.load(datasets_path+"/dev_labels.npy")
    dev_pos_tag_np = np.load(datasets_path+"/dev_pos_tag.npy")
    dev_eojeol_ids_np = np.load(datasets_path+"/dev_eojeol_ids.npy")
    print(f"dev.shape - npy: {dev_npy_np.shape}, label: {dev_label_np.shape}, "
          f"pos_tag: {dev_pos_tag_np.shape}, eojeol_ids: {dev_eojeol_ids_np.shape}")

    model.eval()
    total_data_size = dev_npy_np.shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}

    total_count = 0
    total_eojeol_count = 0
    wrong_count = 0
    vcp_count = 0
    error_idx_list = []
    total_ne_count = 0
    diff_ne_count = 0
    # if search_mode:
    #     while True:
    #         print("input >>>>")
    #         target_text = str(input())
    for data_idx in range(total_data_size):
        inputs = {
            "input_ids": torch.LongTensor([dev_npy_np[data_idx, :, 0]]),
            "attention_mask": torch.LongTensor([dev_npy_np[data_idx, :, 1]]),
            "token_type_ids": torch.LongTensor([dev_npy_np[data_idx, :, 2]]),
            "labels": torch.LongTensor([dev_label_np[data_idx, :]]),
            "eojeol_ids": torch.LongTensor([dev_eojeol_ids_np[data_idx, :]]),
            "pos_tag_ids": torch.LongTensor([dev_pos_tag_np[data_idx, :]])
        }

        # Make Eojeol
        text: List[str] = []
        eojeol_ids = dev_eojeol_ids_np[data_idx, :]
        merge_idx = 0
        for eojeol_cnt in eojeol_ids:
            if 0 >= eojeol_cnt:
                break
            eojeol_tokens = dev_npy_np[data_idx, :, 0][merge_idx:merge_idx + eojeol_cnt]
            merge_idx += eojeol_cnt
            conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
            text.append(conv_eojeol_tokens)
        total_eojeol_count += (len(text) - 1)

        # if 0 < len(target_text):
        #     if target_text.replace(" ", "") not in "".join(text[1:]).replace(" ", ""):
        #         continue

        # Model
        outputs = model(**inputs)
        # loss = outputs.loss
        logits = outputs.logits.detach().cpu().numpy()

        preds = np.array(logits)[0]
        preds = np.argmax(preds, axis=1)
        labels = dev_label_np[data_idx, :]

        # 몇 개가 오류인지
        conv_preds = [ne_ids_to_tag[x] for x in preds]
        conv_labels = [ne_ids_to_tag[x] for x in labels]

        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        for type_name, start, end in get_entities(conv_labels):
            entities_true[type_name].add((start, end))
        for type_name, start, end in get_entities(conv_preds):
            entities_pred[type_name].add((start, end))
        target_names = sorted(set(entities_true.keys()))# | set(entities_pred.keys()))

        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())

            true_type_cnt = len(entities_true_type)
            pred_type_cnt = len(entities_pred_type)

            total_ne_count += true_type_cnt
            diff_ne_count += abs(true_type_cnt - pred_type_cnt)

        # 출력
        pos_ids = dev_pos_tag_np[data_idx, :]
        columns = ["text", "labels", "preds", "pos"]
        row_list = []
        for p_idx in range(len(text)):
            conv_label = ne_ids_to_tag[labels[p_idx]]
            conv_preds = ne_ids_to_tag[preds[p_idx]]
            conv_pos = [pos_ids_to_tag[x] for x in pos_ids[p_idx]]
            row_list.append([text[p_idx], conv_label, conv_preds, conv_pos])
        pd_df = pd.DataFrame(row_list, columns=columns)

        is_wrong_predict = False
        total_count += 1
        for df_idx, df_item in pd_df.iterrows():
            if 0 >= len(df_item["text"]):
                break
            if "VCP" in df_item["pos"]:
                vcp_count += 1
            if ("VCP" in df_item["pos"]) and (pd_df.loc[df_idx-1]["labels"] != pd_df.loc[df_idx-1]["preds"]):
                if not is_wrong_predict:
                    wrong_count += 1
                    error_idx_list.append(data_idx)
                is_wrong_predict = True
            # print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

        print(f"total_count: {total_count}, wrong_count: {wrong_count}, vcp_eojeol_count: {vcp_count}")
        print(f"total_eojeol_count: {total_eojeol_count}, total_ne_count: {total_ne_count}, diff_ne_count: {diff_ne_count}")
        if not is_wrong_predict:
            continue
        else:
            print(" ".join(text))
            print("text\tlabel\tpreds\tPOS")
            for df_idx, df_item in pd_df.iterrows():
                if 0 >= len(df_item["text"]):
                    break
                print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

        # Stop
        # if not search_mode:
            input()
    # Write Error
    print(f"erorr_idx_size: {len(error_idx_list)}")
    save_path = "./split_vcp_eojeol_error_idx.pkl"
    with open(save_path, mode="wb") as write_pkl:
        pickle.dump(error_idx_list, write_pkl)
        print("complete - write: ", save_path)

#===========================================================================
def predict_wordpiece_validation_set(
        model_path: str = "", datasets_path: str = "",
        model_name: str = "",compare_mode: bool = False
):
#===========================================================================
    # load model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ELECTRA_POS_LSTM.from_pretrained(model_path)

    # load data
    dev_npy_np = np.load(datasets_path + "/dev.npy")
    dev_label_np = np.load(datasets_path + "/dev_labels.npy")
    dev_pos_tag_np = np.load(datasets_path + "/dev_pos_tag.npy")
    dev_eojeol_ids_np = np.load(datasets_path + "/dev_eojeol_ids.npy")
    print(f"dev.shape - npy: {dev_npy_np.shape}, label: {dev_label_np.shape}, "
          f"pos_tag: {dev_pos_tag_np.shape}, eojeol_ids: {dev_eojeol_ids_np.shape}")

    model.eval()
    total_data_size = dev_npy_np.shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}

    total_count = 0
    wrong_count = 0
    if compare_mode:
        while True:
            print("input >>>>")
            target_text = str(input())
            for data_idx in range(total_data_size):
                inputs = {
                    "input_ids": torch.LongTensor([dev_npy_np[data_idx, :, 0]]),
                    "attention_mask": torch.LongTensor([dev_npy_np[data_idx, :, 1]]),
                    "token_type_ids": torch.LongTensor([dev_npy_np[data_idx, :, 2]]),
                    # "labels": torch.LongTensor([dev_label_np[data_idx, :]]),
                    # "eojeol_ids": torch.LongTensor([dev_eojeol_ids_np[data_idx, :]]),
                    "pos_tag_ids": torch.LongTensor([dev_pos_tag_np[data_idx, :]])
                }

                text = tokenizer.decode(inputs["input_ids"][0])
                if 0 < len(target_text):
                    if target_text.replace(" ", "") not in "".join(text[1:]).replace(" ", ""):
                        continue

                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                outputs = model(**inputs)

                preds = np.array(outputs)[0]
                labels = dev_label_np[data_idx, :]
                pos_ids = dev_pos_tag_np[data_idx, :]
                columns = ["text", "labels", "preds", "pos"]
                row_list = []
                for p_idx in range(len(preds)):
                    conv_label = ne_ids_to_tag[labels[p_idx]]
                    conv_preds = ne_ids_to_tag[preds[p_idx]]
                    conv_pos = [pos_ids_to_tag[x] for x in pos_ids[p_idx]]
                    row_list.append([tokens[p_idx], conv_label, conv_preds, conv_pos])
                pd_df = pd.DataFrame(row_list, columns=columns)

                is_wrong_predict = False
                total_count += 1
                for df_idx, df_item in pd_df.iterrows():
                    if "[PAD]" == df_item["text"]:
                        break
                    if df_item["labels"] != df_item["preds"]:
                        if not is_wrong_predict:
                            wrong_count += 1
                        is_wrong_predict = True
                    # print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

                print(f"total_count: {total_count}, wrong_count: {wrong_count}")
                if (not is_wrong_predict) and 0 >= len(target_text):
                    continue
                else:
                    print(text)
                    print("text\tlabel\tpreds\tPOS")
                    for df_idx, df_item in pd_df.iterrows():
                        if "[PAD]" == df_item["text"]:
                            break
                        print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

                # Stop
                if not compare_mode:
                    input()

#===========================================================================
def check_origin_concat_vcp_error(
        model_path: str = "", datasets_path: str = "",
        model_name: str = "", search_mode: bool = False
):
#===========================================================================
    # load model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = Electra_Eojeol_Model.from_pretrained(model_path)

    # load data
    dev_npy_np = np.load(datasets_path+"/dev.npy")
    dev_label_np = np.load(datasets_path+"/dev_labels.npy")
    dev_pos_tag_np = np.load(datasets_path+"/dev_pos_tag.npy")
    dev_eojeol_ids_np = np.load(datasets_path+"/dev_eojeol_ids.npy")
    print(f"dev.shape - npy: {dev_npy_np.shape}, label: {dev_label_np.shape}, "
          f"pos_tag: {dev_pos_tag_np.shape}, eojeol_ids: {dev_eojeol_ids_np.shape}")

    model.eval()
    total_data_size = dev_npy_np.shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}

    total_count = 0
    total_eojeol_count = 0
    wrong_count = 0
    vcp_count = 0
    error_idx_list = []
    total_ne_count = 0
    diff_ne_count = 0
    # if search_mode:
    #     while True:
    #         print("input >>>>")
    #         target_text = str(input())
    for data_idx in range(total_data_size):
        inputs = {
            "input_ids": torch.LongTensor([dev_npy_np[data_idx, :, 0]]),
            "attention_mask": torch.LongTensor([dev_npy_np[data_idx, :, 1]]),
            "token_type_ids": torch.LongTensor([dev_npy_np[data_idx, :, 2]]),
            "labels": torch.LongTensor([dev_label_np[data_idx, :]]),
            "eojeol_ids": torch.LongTensor([dev_eojeol_ids_np[data_idx, :]]),
            "pos_tag_ids": torch.LongTensor([dev_pos_tag_np[data_idx, :]])
        }

        # Make Eojeol
        text: List[str] = []
        eojeol_ids = dev_eojeol_ids_np[data_idx, :]
        merge_idx = 0
        for eojeol_cnt in eojeol_ids:
            if 0 >= eojeol_cnt:
                break
            eojeol_tokens = dev_npy_np[data_idx, :, 0][merge_idx:merge_idx + eojeol_cnt]
            merge_idx += eojeol_cnt
            conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
            text.append(conv_eojeol_tokens)
        total_eojeol_count += (len(text) - 1)

        # if 0 < len(target_text):
        #     if target_text.replace(" ", "") not in "".join(text[1:]).replace(" ", ""):
        #         continue

        # Model
        outputs = model(**inputs)
        # loss = outputs.loss
        logits = outputs.logits.detach().cpu().numpy()

        preds = np.array(logits)[0]
        preds = np.argmax(preds, axis=1)
        labels = dev_label_np[data_idx, :]

        # 몇 개가 오류인지
        conv_preds = [ne_ids_to_tag[x] for x in preds]
        conv_labels = [ne_ids_to_tag[x] for x in labels]

        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)

        for type_name, start, end in get_entities(conv_labels):
            entities_true[type_name].add((start, end))
        for type_name, start, end in get_entities(conv_preds):
            entities_pred[type_name].add((start, end))
        target_names = sorted(set(entities_true.keys()))# | set(entities_pred.keys()))

        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())

            true_type_cnt = len(entities_true_type)
            pred_type_cnt = len(entities_pred_type)

            total_ne_count += true_type_cnt
            diff_ne_count += abs(true_type_cnt - pred_type_cnt)

        # 출력
        pos_ids = dev_pos_tag_np[data_idx, :]
        columns = ["text", "labels", "preds", "pos"]
        row_list = []
        for p_idx in range(len(text)):
            conv_label = ne_ids_to_tag[labels[p_idx]]
            conv_preds = ne_ids_to_tag[preds[p_idx]]
            conv_pos = [pos_ids_to_tag[x] for x in pos_ids[p_idx]]
            row_list.append([text[p_idx], conv_label, conv_preds, conv_pos])
        pd_df = pd.DataFrame(row_list, columns=columns)

        is_wrong_predict = False
        total_count += 1
        for df_idx, df_item in pd_df.iterrows():
            if 0 >= len(df_item["text"]):
                break
            if "VCP" in df_item["pos"]:
                vcp_count += 1
            if (df_item["labels"] != df_item["preds"]) and ("VCP" in df_item["pos"]):
                if not is_wrong_predict:
                    wrong_count += 1
                    error_idx_list.append(data_idx)
                is_wrong_predict = True
            # print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

        print(f"total_count: {total_count}, wrong_count: {wrong_count}, vcp_eojeol_count: {vcp_count}")
        print(f"total_eojeol_count: {total_eojeol_count}, total_ne_count: {total_ne_count}, diff_ne_count: {diff_ne_count}")
        # if not is_wrong_predict:
        #     continue
        # else:
        # print(" ".join(text))
        # print("text\tlabel\tpreds\tPOS")
        # for df_idx, df_item in pd_df.iterrows():
        #     if 0 >= len(df_item["text"]):
        #         break
        #     print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

        # Stop
        # if not compare_mode:
        # input()

    # Write Error
    print(f"error_idx_size: {len(error_idx_list)}")
    save_path = "./origin_eojeol_error_idx.pkl"
    with open(save_path, mode="wb") as write_pkl:
        pickle.dump(error_idx_list, write_pkl)
        print("complete - write: ", save_path)

#===========================================================================
def compare_error_idx(
    err_idx_path_1: str = "", err_idx_path_2: str = "",
    model_name: str = "", datasets_path: str = ""
):
#===========================================================================
    # Load Tokenizer
    tokenizer = ElectraTokenizer.from_pretrained(model_name)

    # Load Datasets
    dev_npy_np = np.load(datasets_path+"/dev.npy")
    dev_label_np = np.load(datasets_path+"/dev_labels.npy")
    dev_pos_tag_np = np.load(datasets_path+"/dev_pos_tag.npy")
    dev_eojeol_ids_np = np.load(datasets_path+"/dev_eojeol_ids.npy")
    print(f"dev.shape - npy: {dev_npy_np.shape}, label: {dev_label_np.shape}, "
          f"pos_tag: {dev_pos_tag_np.shape}, eojeol_ids: {dev_eojeol_ids_np.shape}")

    # Load Error Index
    err_idx_list_1 = []
    err_idx_list_2 = []
    with open(err_idx_path_1, mode="rb") as err_file:
        err_idx_list_1 = pickle.load(err_file)
        err_idx_list_1 = sorted(err_idx_list_1)
    with open(err_idx_path_2, mode="rb") as err_file:
        err_idx_list_2 = pickle.load(err_file)
        err_idx_list_2 = sorted(err_idx_list_2)

    duplicated_err = [] # 둘 다 발생하는 ERR
    only_err_1_list = [] # error_list_1에서만 발생
    for idx_1 in err_idx_list_1:
        if idx_1 in err_idx_list_2:
            duplicated_err.append(idx_1)
        else:
            only_err_1_list.append(idx_1)
    only_err_2_list = [] # error_list_2에서만 발생
    for idx_2 in err_idx_list_2:
        if idx_2 not in err_idx_list_1:
            only_err_2_list.append(idx_2)

    print(f"duplicated_err: {len(duplicated_err)} \n {duplicated_err}")
    print(f"only_err_1_list: {len(only_err_1_list)} \n {only_err_1_list}")
    print(f"only_err_2_list: {len(only_err_2_list)} \n {only_err_2_list}")

    return (duplicated_err, only_err_1_list, only_err_2_list)

#===========================================================================
def extract_dev_examples_by_idx(
        model_path: str = "", datasets_path: str = "",
        model_name: str = "", data_idx_list: List[int] = [],
        save_txt_path: str = ""
):
#===========================================================================
    # load model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = Electra_Eojeol_Model.from_pretrained(model_path)

    # load data
    dev_npy_np = np.load(datasets_path + "/dev.npy")
    dev_label_np = np.load(datasets_path + "/dev_labels.npy")
    dev_pos_tag_np = np.load(datasets_path + "/dev_pos_tag.npy")
    dev_eojeol_ids_np = np.load(datasets_path + "/dev_eojeol_ids.npy")
    print(f"dev.shape - npy: {dev_npy_np.shape}, label: {dev_label_np.shape}, "
          f"pos_tag: {dev_pos_tag_np.shape}, eojeol_ids: {dev_eojeol_ids_np.shape}")

    target_deque = deque()
    for idx in data_idx_list:
        target_deque.append(idx)
    print(f"compare size - data_idx_list: {len(data_idx_list)}, target_deque: {len(target_deque)}")

    total_data_size = dev_npy_np.shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}

    target_idx = target_deque.popleft()
    for data_idx in range(total_data_size):
        if 0 >= len(target_deque):
            print("[extract_dev_examples_by_idx] Target dequeue is Empty !")
            break

        if data_idx != target_idx:
            continue

        inputs = {
            "input_ids": torch.LongTensor([dev_npy_np[data_idx, :, 0]]),
            "attention_mask": torch.LongTensor([dev_npy_np[data_idx, :, 1]]),
            "token_type_ids": torch.LongTensor([dev_npy_np[data_idx, :, 2]]),
            "labels": torch.LongTensor([dev_label_np[data_idx, :]]),
            "eojeol_ids": torch.LongTensor([dev_eojeol_ids_np[data_idx, :]]),
            "pos_tag_ids": torch.LongTensor([dev_pos_tag_np[data_idx, :]])
        }

        text: List[str] = []
        eojeol_ids = dev_eojeol_ids_np[data_idx, :]
        merge_idx = 0
        for eojeol_cnt in eojeol_ids:
            eojeol_tokens = dev_npy_np[data_idx, :, 0][merge_idx:merge_idx + eojeol_cnt]
            merge_idx += eojeol_cnt
            conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
            text.append(conv_eojeol_tokens)

        # Model
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()

        preds = np.array(logits)[0]
        preds = np.argmax(preds, axis=1)
        labels = dev_label_np[data_idx, :]
        pos_ids = dev_pos_tag_np[data_idx, :]
        columns = ["text", "labels", "preds", "pos"]
        row_list = []
        for p_idx in range(len(preds)):
            conv_label = ne_ids_to_tag[labels[p_idx]]
            conv_preds = ne_ids_to_tag[preds[p_idx]]
            conv_pos = [pos_ids_to_tag[x] for x in pos_ids[p_idx]]
            row_list.append([text[p_idx], conv_label, conv_preds, conv_pos])
        pd_df = pd.DataFrame(row_list, columns=columns)
        to_tsv_path = save_txt_path+"/"+str(target_idx)+".tsv"
        pd_df.to_csv(to_tsv_path, sep="\t", encoding="utf-8")
        print(f"save - {to_tsv_path}")
        target_idx = target_deque.popleft()

### MAIN
if "__main__" == __name__:
    # '-이다' error check

    model_path = "./old_eojeol_model/model"
    datasets_path = "./old_eojeol_model/npy"
    model_name = "monologg/koelectra-base-v3-discriminator"
    # check_origin_concat_vcp_error(model_path=model_path,
    #                               datasets_path=datasets_path,
    #                               model_name=model_name,
    #                               search_mode=False)

    # Eojeol Validation Set
    model_path = "./split_vcp_model/model"
    datasets_path = "./split_vcp_model/npy"
    model_name = "monologg/koelectra-base-v3-discriminator"
    predict_eojeol_validation_set(model_path=model_path,
                                  datasets_path=datasets_path,
                                  model_name=model_name,
                                  search_mode=True)

    # Wordpiece Validation Set
    model_path = "./wordpiece_model/model"
    datasets_path = "./wordpiece_model/npy"
    model_name = "monologg/koelectra-base-v3-discriminator"
    # predict_wordpiece_validation_set(model_path=model_path,
    #                                  datasets_path=datasets_path,
    #                                  model_name=model_name,
    #                                  compare_mode=True)


    # STOP
    exit()

    err_file_path_1 = "./origin_eojeol_error_idx.pkl"
    err_file_path_2 = "./split_vcp_eojeol_error_idx.pkl"
    model_name = "monologg/koelectra-base-v3-discriminator"
    datasets_path = "./wordpiece_model/npy"
    duplicated_list, only_err_1_list, only_err_2_list = compare_error_idx(err_idx_path_1=err_file_path_1,
                                                                          err_idx_path_2=err_file_path_2,
                                                                          model_name=model_name,
                                                                          datasets_path=datasets_path)

    # duplicated
    dup_save_path_name = "./extract_data/duplicated/origin"
    extract_dev_examples_by_idx(model_path="./old_eojeol_model/model",
                                model_name="monologg/koelectra-base-v3-discriminator",
                                datasets_path="./old_eojeol_model/npy",
                                data_idx_list=duplicated_list, save_txt_path=dup_save_path_name)

    dup_save_path_name = "./extract_data/duplicated/split_vcp"
    extract_dev_examples_by_idx(model_path="./split_vcp_model/model",
                                model_name="monologg/koelectra-base-v3-discriminator",
                                datasets_path="./split_vcp_model/npy",
                                data_idx_list=duplicated_list, save_txt_path=dup_save_path_name)

    # 기본 어절 모델에만 있는 오류
    dup_save_path_name = "./extract_data/origin_eojeol/origin"
    extract_dev_examples_by_idx(model_path="./old_eojeol_model/model",
                                model_name="monologg/koelectra-base-v3-discriminator",
                                datasets_path="./old_eojeol_model/npy",
                                data_idx_list=only_err_1_list, save_txt_path=dup_save_path_name)

    dup_save_path_name = "./extract_data/origin_eojeol/split_vcp"
    extract_dev_examples_by_idx(model_path="./split_vcp_model/model",
                                model_name="monologg/koelectra-base-v3-discriminator",
                                datasets_path="./split_vcp_model/npy",
                                data_idx_list=only_err_1_list, save_txt_path=dup_save_path_name)
    
    # 서술격조사 분리 모델에만 있는 오류
    dup_save_path_name = "./extract_data/split_vcp/origin"
    extract_dev_examples_by_idx(model_path="./old_eojeol_model/model",
                                model_name="monologg/koelectra-base-v3-discriminator",
                                datasets_path="./old_eojeol_model/npy",
                                data_idx_list=only_err_2_list, save_txt_path=dup_save_path_name)

    dup_save_path_name = "./extract_data/split_vcp/split_vcp"
    extract_dev_examples_by_idx(model_path="./split_vcp_model/model",
                                model_name="monologg/koelectra-base-v3-discriminator",
                                datasets_path="./split_vcp_model/npy",
                                data_idx_list=only_err_2_list, save_txt_path=dup_save_path_name)