import torch
import numpy as np
import pandas as pd

from model.electra_lstm_crf import ELECTRA_POS_LSTM
from model.electra_eojeol_model import Electra_Eojeol_Model
from transformers import ElectraTokenizer

from ner_def import ETRI_TAG

## MAIN
if "__main__" == __name__:
    print(f"[test_ner][__main__] MAIN !")

    # Data Load
    root_path = "./corpus/npy/eojeol_electra"
    train_ids_data = np.load(root_path + "/train.npy")
    train_pos_data = np.load(root_path + "/train_pos_tag.npy")
    train_labels_data = np.load(root_path + "/train_labels.npy")
    train_eojeol_ids = np.load(root_path + "/train_eojeol_ids.npy")
    train_token_seq_len = np.load(root_path + "/train_token_seq_len.npy")
    print(train_token_seq_len.shape)

    dev_ids_data = np.load(root_path + "/dev.npy")
    dev_pos_data = np.load(root_path + "/dev_pos_tag.npy")
    dev_labels_data = np.load(root_path + "/dev_labels.npy")
    dev_eojeol_ids = np.load(root_path + "/dev_eojeol_ids.npy")
    dev_token_seq_len = np.load(root_path + "/dev_token_seq_len.npy")
    print(dev_token_seq_len.shape)

    test_ids_data = np.load(root_path + "/test.npy")
    test_pos_data = np.load(root_path + "/test_pos_tag.npy")
    test_labels_data = np.load(root_path + "/test_labels.npy")
    test_eojeol_ids = np.load(root_path + "/test_eojeol_ids.npy")
    test_token_seq_len = np.load(root_path + "/test_token_seq_len.npy")
    print(test_token_seq_len.shape)

    total_ids_data = np.vstack([train_ids_data, dev_ids_data, test_ids_data])
    total_pos_data = np.vstack([train_pos_data, dev_pos_data, test_pos_data])
    total_labels_data = np.vstack([train_labels_data, dev_labels_data, test_labels_data])
    total_eojeol_ids = np.vstack([train_eojeol_ids, dev_eojeol_ids, test_eojeol_ids])
    total_token_seq_len = np.vstack([train_token_seq_len, dev_token_seq_len, test_token_seq_len])

    # Tokenizer
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    # Eojeol Model
    checkpoint_path = "./test_model"
    model = Electra_Eojeol_Model.from_pretrained(checkpoint_path)
    model.eval()

    total_data_size = total_ids_data.shape[0]
    ids_to_tag = {v: k for k, v in ETRI_TAG.items()}

    while True:
        user_input = str(input())
        target_tokens = [tokenizer(user_input, padding='max_length', truncation=True, max_length=128)["input_ids"]]

        for data_idx in range(total_data_size):
            if 0 == (data_idx % 10000):
                print(f"[test_ner][__main__] {data_idx} is processing...")

            inputs = {
                "input_ids": torch.LongTensor([total_ids_data[data_idx, :, 0]]),
                "attention_mask": torch.LongTensor([total_ids_data[data_idx, :, 1]]),
                "token_type_ids": torch.LongTensor([total_ids_data[data_idx, :, 2]]),
                # "labels": torch.LongTensor([total_labels_data[data_idx, :]]),
                "token_seq_len": torch.LongTensor([total_token_seq_len[data_idx, :]]),
                "pos_tag_ids": torch.LongTensor([total_pos_data[data_idx, :]]),
                "eojeol_ids": torch.LongTensor([total_eojeol_ids[data_idx, :]])
            }

            if target_tokens != inputs["input_ids"].tolist():
                continue
            else:
                print("FIND !")

            outputs = model(**inputs)

            text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            preds = np.array(outputs)[0]
            labels = total_labels_data[data_idx, :]

            columns = ["text", "label", "preds"]
            rows_list = []
            for p_idx in range(len(preds)):
                conv_label = ids_to_tag[labels[p_idx]]
                conv_preds = ids_to_tag[preds[p_idx]]
                rows_list.append([text[p_idx], conv_label, conv_preds])
            pd_df = pd.DataFrame(rows_list, columns=columns)

            print("text\tlabel\tpreds")
            for df_idx, df_item in pd_df.iterrows():
                print(df_item["text"], df_item["label"], df_item["preds"])