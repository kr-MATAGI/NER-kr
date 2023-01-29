import numpy as np
import torch
import re

from torch.utils.data import Dataset
from typing import List, Dict

### ELECTRA, BERT
#===============================================================
class NER_POS_Dataset(Dataset):
#===============================================================
    def __init__(
            self,
            item_dict: Dict[str, torch.Tensor]
    ):
        self.input_ids = item_dict["input_ids"]
        self.attention_mask = item_dict["attention_mask"]
        self.token_type_ids = item_dict["token_type_ids"]
        self.labels = item_dict["label_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label_ids": self.labels[idx],
            # "pos_ids": self.pos_ids[idx]
        }

        return items

#===============================================================
class NER_Eojeol_Datasets(Dataset):
#===============================================================
    def __init__(
        self,
        token_data: np.ndarray, labels: np.ndarray,
        pos_tag_ids: np.ndarray, eojeol_ids: np.ndarray#, token_seq_len: np.ndarray,
    ):
        # unit: wordpiece token
        self.input_ids = token_data[:][:, :, 0]
        self.attention_mask = token_data[:][:, :, 1]
        self.token_type_ids = token_data[:][:, :, 2]
        #self.token_seq_len = token_seq_len

        # unit: split_vcp
        self.labels = labels
        self.pos_tag_ids = pos_tag_ids
        self.eojeol_ids = eojeol_ids

        # convert numpy to tensor
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)
        #self.token_seq_len = torch.tensor(self.token_seq_len, dtype=torch.long)

        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.pos_tag_ids = torch.tensor(self.pos_tag_ids, dtype=torch.long)
        self.eojeol_ids = torch.tensor(self.eojeol_ids, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            #"token_seq_len": self.token_seq_len[idx],
            "labels": self.labels[idx],
            "pos_tag_ids": self.pos_tag_ids[idx],
            "eojeol_ids": self.eojeol_ids[idx],
            #"ls_ids": self.ls_ids[idx]
        }

        return items

#===============================================================
class SpanNERDataset(Dataset):
#===============================================================
    def __init__(
            self,
            data: np.ndarray, label_ids: np.ndarray,
            all_span_idx:np.ndarray, all_span_len: np.ndarray,
            real_span_mask: np.ndarray, span_only_label: np.ndarray,
            pos_ids: np.ndarray
    ):
        self.input_ids = torch.LongTensor(data[:][:, :, 0])
        self.attn_mask = torch.LongTensor(data[:][:, :, 1])
        self.token_type_ids = torch.LongTensor(data[:][:, :, 2])
        self.label_ids = torch.LongTensor(label_ids)

        self.all_span_idx = torch.LongTensor(all_span_idx)
        self.all_span_len = torch.LongTensor(all_span_len)
        self.real_span_mask = torch.LongTensor(real_span_mask)
        self.span_only_label = torch.LongTensor(span_only_label)

        self.pos_ids = torch.LongTensor(pos_ids)

    def __len__(self):
        return self.input_ids.size()[0]

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "label_ids": self.label_ids[idx],

            "all_span_idx": self.all_span_idx[idx],
            "all_span_len": self.all_span_len[idx],
            "real_span_mask": self.real_span_mask[idx],
            "span_only_label": self.span_only_label[idx],

            "pos_ids": self.pos_ids[idx],
        }

        return items