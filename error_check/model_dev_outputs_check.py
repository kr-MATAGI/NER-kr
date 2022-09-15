import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd

from transformers import ElectraModel, ElectraTokenizer
from model.electra_eojeol_model import Electra_Eojeol_Model
from utils.tag_def import ETRI_TAG

#===========================================================================
def predict_validation_set(model_path: str = "", datasets_path: str = "", model_name: str = ""):
#===========================================================================
    # load model
    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = Electra_Eojeol_Model.from_pretrained(model_path)

    # load data
    dev_npy_np = np.load(datasets_path+"/dev.npy")
    dev_label_np = np.load(datasets_path+"/dev_labels.npy")
    dev_pos_tag_np = np.load(datasets_path+"/dev_pos_tag.npy")
    dev_eojeol_ids_np = np.load(datasets_path+"/dev_eojeol_ids.npy")
    print(f"dev.shape - npy: {dev_npy_np.shape}, label: {dev_label_np.shape}, pos_tag: {dev_pos_tag_np.shape},"
          f"eojeol_ids: {dev_eojeol_ids_np.shape}")

    model.eval()
    total_data_size = dev_npy_np.shape[0]
    ne_ids_to_tag = {v: k for k, v in ETRI_TAG.items()}
    for data_idx in range(total_data_size):
        inputs = {
            "input_ids": torch.LongTensor([dev_npy_np[data_idx, :, 0]]),
            "attention_mask": torch.LongTensor([dev_npy_np[data_idx, :, 1]]),
            "token_type_ids": torch.LongTensor([dev_npy_np[data_idx, :, 2]]),
            "labels": torch.LongTensor([dev_label_np[data_idx, :]]),
            "eojeol_ids": torch.LongTensor([dev_eojeol_ids_np[data_idx, :]]),
            "pos_tag_ids": torch.LongTensor([dev_pos_tag_np[data_idx, :]])
        }

        outputs = model(**inputs)
        # loss = outputs.loss
        logits = outputs.logits.detach().cpu().numpy()

        text = tokenizer.decode(inputs["input_ids"][0]).split(" ")
        print(text)
        preds = np.array(logits)[0]
        preds = np.argmax(preds, axis=1)
        labels = dev_label_np[data_idx, :]
        columns = ["text", "labels", "preds"]
        row_list = []
        for p_idx in range(len(preds)):
            conv_label = ne_ids_to_tag[labels[p_idx]]
            conv_preds = ne_ids_to_tag[preds[p_idx]]
            row_list.append([text[p_idx], conv_label, conv_preds])
        pd_df = pd.DataFrame(row_list, columns=columns)

        print("text\tlabel\tpreds")
        for df_idx, df_item in pd_df.iterrows():
            print(df_item["text"], df_item["labels"], df_item["preds"])

        # stop
        input()

### MAIN
if "__main__" == __name__:
    model_path = "./eojeol_model/model"
    datasets_path = "./eojeol_model/npy"
    model_name = "monologg/koelectra-base-v3-discriminator"
    predict_validation_set(model_path=model_path,
                           datasets_path=datasets_path,
                           model_name=model_name)