import torch
import torch.nn as nn
import numpy as np
from transformers import ElectraPreTrainedModel, ElectraModel, ElectraConfig
from transformers.modeling_outputs import TokenClassifierOutput
from model.transformer_encoder import Trans_Encoder, Enc_Config

#===============================================================
class ELECTRA_CNNBiF_Model(ElectraPreTrainedModel):
#===============================================================
    #===================================
    def __init__(self, config):
    #===================================
        super(ELECTRA_CNNBiF_Model, self).__init__(config)
        # init
        self.dropout_rate = 0.1

        # default config
        self.config = config

        # Transformer Encoder Config
        self.d_model_size = config.hidden_size
        self.transformer_config = Enc_Config(vocab_size_or_config_json_file=config.vocab_size)
        self.transformer_config.hidden_size = self.d_model_size

        # KoELECTRA
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

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
        self.ne_classifier = nn.Linear(config.hidden_size, config.num_labels)
        # LS - Classifier
        self.ls_classifier = nn.Linear(config.hidden_size, 2)

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

        # Transformer Encoder
        enc_outputs = self.trans_encoder(eojeol_tensor, eojeol_attn_mask)
        enc_outputs = enc_outputs[-1]

        # NER Sequence Labeling
        logits = self.ne_classifier(enc_outputs)
        ner_loss = None
        if labels is not None:
            ner_loss_fct = nn.CrossEntropyLoss()
            ner_loss = ner_loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

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

        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits
        )

if "__main__" == __name__:
    config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    print(config)