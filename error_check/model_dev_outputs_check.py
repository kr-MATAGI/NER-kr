import numpy as np
import torch
from typing import List
import pandas as pd

import sys
sys.path.append("C:/Users/MATAGI/Desktop/Git/NER_Private")
from transformers import ElectraModel, ElectraTokenizer
from model.electra_eojeol_model import Electra_Eojeol_Model
from model.electra_lstm_crf import ELECTRA_POS_LSTM
from utils.tag_def import ETRI_TAG, NIKL_POS_TAG

#===========================================================================
def predict_eojeol_validation_set(
        model_path: str = "", datasets_path: str = "",
        model_name: str = "", compare_mode: bool = False
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
    wrong_count = 0
    # if compare_mode:
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
            eojeol_tokens = dev_npy_np[data_idx, :, 0][merge_idx:merge_idx + eojeol_cnt]
            merge_idx += eojeol_cnt
            conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
            text.append(conv_eojeol_tokens)

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
        pos_ids = dev_pos_tag_np[data_idx, :]
        columns = ["text", "labels", "preds", "pos"]
        row_list = []
        for p_idx in range(len(preds)):
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
            if df_item["labels"] != df_item["preds"]:
                if not is_wrong_predict:
                    wrong_count += 1
                is_wrong_predict = True
            # print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

        print(f"total_count: {total_count}, wrong_count: {wrong_count}")
        # if (not is_wrong_predict): #and 0 >= len(target_text):
        #     continue
        # else:
        print(" ".join(text))
        print("text\tlabel\tpreds\tPOS")
        for df_idx, df_item in pd_df.iterrows():
            if 0 >= len(df_item["text"]):
                break
            print(df_item["text"], df_item["labels"], df_item["preds"], df_item["pos"])

        # Stop
        if not compare_mode:
            input()

#===========================================================================
def predict_wordpiece_validation_set(model_path: str = "", datasets_path: str = "", model_name: str = "",
                                     compare_mode: bool = False):
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

### MAIN
if "__main__" == __name__:
    model_path = "./old_eojeol_model/model"
    datasets_path = "./old_eojeol_model/npy"
    model_name = "monologg/koelectra-base-v3-discriminator"

    # Eojeol Validation Set
    predict_eojeol_validation_set(model_path=model_path,
                                  datasets_path=datasets_path,
                                  model_name=model_name,
                                  compare_mode=False)

    # Wordpiece Validation Set
    model_path = "./wordpiece_model/model"
    datasets_path = "./wordpiece_model/npy"
    model_name = "monologg/koelectra-base-v3-discriminator"
    # predict_wordpiece_validation_set(model_path=model_path,
    #                                  datasets_path=datasets_path,
    #                                  model_name=model_name,
    #                                  compare_mode=True)