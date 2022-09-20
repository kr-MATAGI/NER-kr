import pickle
import numpy as np
import pandas as pd
import os
from transformers import ElectraTokenizer
from utils.tag_def import ETRI_TAG, NIKL_POS_TAG

from typing import List

#============================================================
def check_make_nn_pos_pattern(
        data_root_path: str = "", model_name: str = "",
        display_mode: bool = False, check_limit_size: int = 10
):
#============================================================
    # Load Model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)

    # Load Train Datasets
    train_np = np.load(data_root_path+"/train.npy")
    train_input_ids = train_np[:, :, 0]
    train_attention_mask = train_np[:, :, 1]
    train_token_type_ids = train_np[:, :, 2]
    print(f"train - input_ids: {train_input_ids.shape}, "
          f"attn_mask: {train_attention_mask.shape}, "
          f"token_type_ids: {train_token_type_ids.shape}")

    train_pos_tag_np = np.load(data_root_path+"/train_pos_tag.npy")
    train_eojeol_ids_np = np.load(data_root_path+"/train_eojeol_ids.npy")
    train_labels_np = np.load(data_root_path+"/train_labels.npy")
    print(f"train - pos_tag: {train_pos_tag_np.shape}, "
          f"eojeol_ids: {train_eojeol_ids_np.shape}, "
          f"labels_ids: {train_labels_np.shape}")

    total_data_size = train_input_ids.shape[0]
    pos_ids_to_tag = {k: v for k, v in NIKL_POS_TAG.items()}
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}

    pattern_dict = {} # key: pattern, value: (word, ne_label)

    for data_idx in range(total_data_size):
        if 0 == (data_idx % 1000):
            print(f"{data_idx} is Processing...")

        # Make Eojeol
        text: List[str] = []
        eojeol_ids = train_eojeol_ids_np[data_idx, :]
        merge_idx = 0
        for eojeol_cnt in eojeol_ids:
            eojeol_tokens = train_input_ids[data_idx][merge_idx:merge_idx + eojeol_cnt]
            merge_idx += eojeol_cnt
            conv_eojeol_tokens = tokenizer.decode(eojeol_tokens)
            text.append(conv_eojeol_tokens)
        text = [x for x in text if 0 < len(x)]
        labels = train_labels_np[data_idx, :]
        labels = [ne_ids_to_tag[x] for x in labels]
        pos_ids = train_pos_tag_np[data_idx, :]
        pos_ids = [[pos_ids_to_tag[x] for x in pos_row] for pos_row in pos_ids]

        columns = ["text", "labels", "pos"]
        row_list = []
        for p_idx in range(len(text)):
            row_list.append([text[p_idx], labels[p_idx], pos_ids[p_idx]])
        pd_df = pd.DataFrame(row_list, columns=columns)

        if display_mode:
            for df_idx, df_item in pd_df.iterrows():
                print(df_item["text"], df_item["labels"], df_item["pos"])
            input()

        # check pattern
        front_morp_target = ["NNG", "NNP", "CONCAT_NN"]
        for df_idx, df_item in pd_df.iterrows():
            valid_morp_list = [x for x in df_item["pos"] if x != "O"]
            if df_item["pos"][0] in front_morp_target and check_limit_size >= len(valid_morp_list):
                merge_morp_key = "+".join(valid_morp_list)
                if merge_morp_key in pattern_dict.keys():
                    pattern_dict[merge_morp_key].append((df_item["text"], df_item["labels"]))
                else:
                    pattern_dict[merge_morp_key] = [(df_item["text"], df_item["labels"])]

    # Write Files
    dict_keys = sorted(pattern_dict.keys(), key=lambda x: len(x))
    for key in dict_keys:
        with open("./results/"+key+".txt", mode="w", encoding="utf-8") as new_file:
            values = pattern_dict[key]
            for v in values:
                new_file.write(v[0]+"\t"+v[1]+"\n")

### MAIN
if "__main__" == __name__:
    data_root_path = "../corpus/npy/eojeol_not_split_electra"
    model_name = "monologg/koelectra-base-v3-discriminator"
    check_make_nn_pos_pattern(data_root_path=data_root_path,
                              model_name=model_name,
                              display_mode=False)