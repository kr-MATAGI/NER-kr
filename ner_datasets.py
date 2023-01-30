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
        self.input_ids = torch.tensor(item_dict["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(item_dict["attention_mask"], dtype=torch.long)
        self.token_type_ids = torch.tensor(item_dict["token_type_ids"], dtype=torch.long)
        self.labels = torch.tensor(item_dict["label_ids"], dtype=torch.long)

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
class SpanNERDataset(Dataset):
#===============================================================
    def __init__(
            self,
            item_dict: Dict[str, np.ndarray]
    ):
        self.input_ids = torch.tensor(item_dict["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(item_dict["attention_mask"], dtype=torch.long)
        self.token_type_ids = torch.tensor(item_dict["token_type_ids"], dtype=torch.long)
        self.labels = torch.tensor(item_dict["label_ids"], dtype=torch.long)

        self.all_span_idx = torch.tensor(item_dict["all_span_idx"], dtype=torch.long)
        self.all_span_len = torch.tensor(item_dict["all_span_len"], dtype=torch.long)
        self.real_span_mask = torch.tensor(item_dict["real_span_mask"], dtype=torch.long)
        self.span_only_label = torch.tensor(item_dict["span_only_label"], dtype=torch.long)

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
        }

        return items