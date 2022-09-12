import numpy as np
import torch
from torch.utils.data import Dataset

### ELECTRA, BERT
#===============================================================
class NER_POS_Dataset(Dataset):
#===============================================================
    def __init__(
            self,
            data: np.ndarray, labels: np.ndarray,
            pos_tag_ids: np.ndarray, eojeol_ids: np.ndarray#, entity_ids: np.ndarray, toekn_seq_len: np.ndarray
    ):
        self.input_ids = data[:][:, :, 0]
        self.attention_mask = data[:][:, :, 1]
        self.token_type_ids = data[:][:, :, 2]
        self.labels = labels
        #self.token_seq_len = toekn_seq_len
        self.pos_tag_ids = pos_tag_ids
        self.eojeol_ids = eojeol_ids
        #self.entity_ids = entity_ids

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(self.token_type_ids, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        #self.token_seq_len = torch.tensor(self.token_seq_len, dtype=torch.long)
        self.pos_tag_ids = torch.tensor(self.pos_tag_ids, dtype=torch.long)
        self.eojeol_ids = torch.tensor(self.eojeol_ids, dtype=torch.long)
        #self.entity_ids = torch.tensor(self.entity_ids, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        items = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
            #"token_seq_len": self.token_seq_len[idx],
            "pos_tag_ids": self.pos_tag_ids[idx],
            "eojeol_ids": self.eojeol_ids[idx],
            #"entity_ids": self.entity_ids[idx]
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

        # unit: eojeol
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