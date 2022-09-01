import copy

import torch
import torch.nn as nn
import numpy as np
from transformers import ElectraPreTrainedModel, ElectraModel, ElectraConfig
from transformers.modeling_outputs import TokenClassifierOutput
from model.transformer_encoder import Trans_Encoder, Enc_Config

from model.crf_layer import CRF

#===============================================================
class ELECTRA_CNNBiF_Model(ElectraPreTrainedModel):
#===============================================================
    #===================================
    def __init__(self, config):
    #===================================
        super(ELECTRA_CNNBiF_Model, self).__init__(config)
        # init
        self.dropout_rate = 0.1
        self.pos_embed_dim = 102

        # default config
        self.config = config

        # Transformer Encoder Config
        self.d_model_size = config.hidden_size + (self.pos_embed_dim * 4)
        self.transformer_config = Enc_Config(vocab_size_or_config_json_file=config.vocab_size)
        self.transformer_config.hidden_size = self.d_model_size

        # KoELECTRA
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # POS Embedding
        self.eojeol_pos_embedding_1 = nn.Embedding(config.num_pos_labels, self.pos_embed_dim)
        self.eojeol_pos_embedding_2 = nn.Embedding(config.num_pos_labels, self.pos_embed_dim)
        self.eojeol_pos_embedding_3 = nn.Embedding(config.num_pos_labels, self.pos_embed_dim)
        self.eojeol_pos_embedding_4 = nn.Embedding(config.num_pos_labels, self.pos_embed_dim)

        # Transformer Encoder
        self.trans_encoder = Trans_Encoder(self.transformer_config)

        # Eojeol Boundary Embedding
        self.eojeol_boundary_embedding = nn.Embedding(config.max_seq_len, config.max_eojeol_len)

        # CNNBiF
        self.cnn_bi_f = nn.Conv1d(
            in_channels=config.max_eojeol_len, out_channels=config.max_eojeol_len,
            kernel_size=2, padding=1
        )

        # NE - Classifier
        self.ne_classifier = nn.Linear(self.d_model_size, config.num_labels)
        # CRF
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # LS - Classifier
        self.ls_classifier = nn.Linear(config.hidden_size + 1, 2)

        self.post_init()

    #===================================
    def _make_one_hot_embedding(
            self,
            last_hidden,
            eojeol_ids
    ):
    #===================================
        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        new_eojeol_info_matrix = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for batch_idx in range(batch_size):
            cur_idx = 0
            eojeol_info_matrix = torch.zeros(max_seq_len, dtype=torch.long)
            for eojeol_idx, eojeol_tok_cnt in enumerate(eojeol_ids[batch_idx]):
                tok_cnt = eojeol_tok_cnt.detach().cpu().item()
                if 0 == tok_cnt:
                    break
                for _ in range(tok_cnt):
                    if max_seq_len <= cur_idx:
                        break
                    eojeol_info_matrix[cur_idx] = eojeol_idx
                    cur_idx += 1
            new_eojeol_info_matrix[batch_idx] = eojeol_info_matrix

        new_eojeol_info_matrix = new_eojeol_info_matrix.to(device)
        eojeol_boundary_embed = self.eojeol_boundary_embedding(new_eojeol_info_matrix)

        return eojeol_boundary_embed

    #===================================
    def _make_eojeol_tensor(
            self,
            last_hidden,
            eojeol_ids,
            eojeol_bound_embd,
            max_eojeol_len=50
    ):
    #===================================
        '''
            last_hidden.shape: [batch_size, token_seq_len, hidden_size]
            token_seq_len: [batch_size, ]
            pos_ids: [batch_size, eojeol_seq_len, pos_tag_size]
            eojeol_ids: [batch_size, eojeol_seq_len]
        '''
        batch_size, max_seq_len, hidden_size = last_hidden.size()
        device = last_hidden.device
        all_eojeol_attention_mask = torch.zeros(batch_size, max_eojeol_len)

        matmul_out_embed = eojeol_bound_embd @ last_hidden  # [64, 50, 768] = [batch, max_eojeol, hidden]
        for batch_idx in range(batch_size):
            valid_eojeol_cnt = 0
            for eojeol_idx, eojeol_token_cnt in enumerate(eojeol_ids[batch_idx]):
                if 0 == eojeol_token_cnt:
                    break
                valid_eojeol_cnt += 1

            # end, eojeol loop
            eojeol_attention_mask = torch.zeros(max_eojeol_len)
            for i in range(valid_eojeol_cnt):
                eojeol_attention_mask[i] = 1
            all_eojeol_attention_mask[batch_idx] = eojeol_attention_mask
        # end, batch_loop
        return matmul_out_embed, all_eojeol_attention_mask.to(device)

    #===================================
    def forward(
            self,
            input_ids, attention_mask, token_type_ids, token_seq_len=None, # Unit: Token
            labels=None, pos_tag_ids=None, eojeol_ids=None, ls_ids=None # Unit: Eojeol
    ):
    #===================================
        electra_outputs = self.electra(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)
        electra_last_hidden = electra_outputs.last_hidden_state

        # eojeol embed : [batch_size, max_seq_len, max_eojeol_len]
        eojeol_boundary_embed = self._make_one_hot_embedding(last_hidden=electra_last_hidden, eojeol_ids=eojeol_ids)

        # matmul
        eojeol_boundary_embed = eojeol_boundary_embed.transpose(1, 2)
        eojeol_tensor, eojeol_attn_mask = self._make_eojeol_tensor(last_hidden=electra_last_hidden,
                                                                   #pos_tag_ids=pos_tag_ids,
                                                                   eojeol_ids=eojeol_ids,
                                                                   eojeol_bound_embd=eojeol_boundary_embed,
                                                                   max_eojeol_len=self.config.max_eojeol_len)
        eojeol_attn_mask_copy = copy.deepcopy(eojeol_attn_mask)
        eojeol_attn_mask = eojeol_attn_mask.unsqueeze(1).unsqueeze(2)
        eojeol_attn_mask = eojeol_attn_mask.to(dtype=next(self.parameters()).dtype)
        eojeol_attn_mask = (1.0 - eojeol_attn_mask) * -10000.0

        # POS Embedding : [batch, eojeol_max_len] -> [batch, eojeol_max_len, pos_embed]
        eojeol_pos_1 = pos_tag_ids[:, :, 0]
        eojeol_pos_2 = pos_tag_ids[:, :, 1]
        eojeol_pos_3 = pos_tag_ids[:, :, 2]
        eojeol_pos_4 = pos_tag_ids[:, :, 3]

        eojeol_pos_1 = self.eojeol_pos_embedding_1(eojeol_pos_1)
        eojeol_pos_2 = self.eojeol_pos_embedding_2(eojeol_pos_2)
        eojeol_pos_3 = self.eojeol_pos_embedding_3(eojeol_pos_3)
        eojeol_pos_4 = self.eojeol_pos_embedding_4(eojeol_pos_4)

        concat_eojeol_pos = torch.concat([eojeol_pos_1, eojeol_pos_2, eojeol_pos_3, eojeol_pos_4], dim=-1)
        eojeol_pos_concat = torch.concat([eojeol_tensor, concat_eojeol_pos], dim=-1)

        # Transformer Encoder
        enc_outputs = self.trans_encoder(eojeol_pos_concat, eojeol_attn_mask)
        enc_outputs = enc_outputs[-1]

        # NER Sequence Labeling
        logits = self.ne_classifier(enc_outputs)
        ner_loss = None
        if labels is not None:
            # ner_loss_fct = nn.CrossEntropyLoss()
            # ner_loss = ner_loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            # CRF
            ner_loss, sequence_of_tags = self.crf(emissions=logits, tags=labels, reduction="mean",
                                                  mask=eojeol_attn_mask_copy.bool()), \
                                         self.crf.decode(logits, mask=eojeol_attn_mask_copy.bool())
            ner_loss *= -1
        else:
            sequence_of_tags = self.crf.decode(logits)

        # CNNBiF
        cnn_bi_f_outputs = self.cnn_bi_f(eojeol_tensor) # [batch, kernel, hidden-1]
        cnn_bi_f_outputs = self.ls_classifier(cnn_bi_f_outputs) # [batch, eojeol, 2]

        # Get Loss
        ls_loss = None
        if ls_ids is not None:
            # LS_ids : ["L", "S"]
            ls_loss_fct = nn.CrossEntropyLoss()
            ls_loss = ls_loss_fct(cnn_bi_f_outputs.view(-1, 2), ls_ids.view(-1))

        total_loss = None
        if (labels is not None) and (ls_ids is not None):
            total_loss = ner_loss + ls_loss
        elif ls_ids is None:
            total_loss = ner_loss

        # return TokenClassifierOutput(
        #     loss=total_loss,
        #     logits=logits
        # )
        if labels is not None:
            return total_loss, sequence_of_tags
        else:
            return sequence_of_tags

if "__main__" == __name__:
    config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    print(config)