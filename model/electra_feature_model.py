import torch
import copy
import torch.nn as nn

from transformers import ElectraModel, ElectraPreTrainedModel
from model.crf_layer import CRF
from model.transformer_encoder import Trans_Encoder

#===============================================================
class Electra_Feature_Model(ElectraPreTrainedModel):
#===============================================================
    def __init__(self, config):
        # init
        super(Electra_Feature_Model, self).__init__(config)

        self.max_seq_len = config.max_seq_len
        self.max_eojeol_len = config.max_eojeol_len

        self.num_ne_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels

        self.pos_embed_dim = 128
        self.entity_embed_dim = 120
        # self.ffn_1 = 1024

        self.dropout_rate = 0.1

        # for Transformer Encoder Config
        self.d_model_size = config.hidden_size + (self.pos_embed_dim * 3) + self.max_eojeol_len
        self.t_enc_config = copy.deepcopy(config)
        self.t_enc_config.num_hidden_layers = 4
        self.t_enc_config.hidden_size = self.d_model_size
        self.t_enc_config.ff_dim = self.d_model_size
        self.t_enc_config.act_fn = "gelu"
        self.t_enc_config.dropout_prob = 0.1
        self.t_enc_config.num_heads = 12

        # Pre-trained Model
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # POS Embedding
        self.pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_dim)
        self.pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_dim)
        self.pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_dim)

        # Eojeol Boundary Embedding
        self.eojeol_embedding = nn.Embedding(self.max_seq_len, self.max_eojeol_len)

        # Entity Embedding
        # self.entity_embedding = nn.Embedding(self.max_seq_len, self.entity_embed_dim)

        self.eojeol_entity_embedding = nn.Embedding(self.max_seq_len, self.entity_embed_dim)

        # FFN_1
        # self.ffn_1 = nn.Linear()

        # Transformer Encoder
        self.trans_encoder = Trans_Encoder(self.t_enc_config)

        # FFN_2 (for CRF)
        self.ffn_2 = nn.Linear(self.d_model_size, config.num_labels)

        # CRF
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    #===================================
    def _make_eojeol_entity_embed(self, eojeol_ids, entity_ids):
    #===================================
        batch_size, seq_len = eojeol_ids.size() # [64, 128]

        device = eojeol_ids.device
        eojeol_ids = eojeol_ids.detach().cpu()
        entity_ids = entity_ids.detach().cpu()
        new_tensor = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for batch_idx in range(batch_size):
            for idx, (eojeol, entity) in enumerate(zip(eojeol_ids[batch_idx], entity_ids[batch_idx])):
                if 0 != eojeol:
                    new_tensor[batch_idx][idx] = 1
                if 0 != entity:
                    new_tensor[batch_idx][idx] = 2
        return new_tensor.to(device)

    #===================================
    def forward(
            self,
            input_ids, attention_mask, token_type_ids,
            token_seq_len=None, labels=None, pos_tag_ids=None,
            eojeol_ids=None, entity_ids=None
    ):
    #===================================
        # PLM
        plm_outputs = self.electra(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)

        # [batch_size, max_seq_len, hidden_size]
        plm_last_hidden = plm_outputs.last_hidden_state

        # POS
        token_pos_1 = pos_tag_ids[:, :, 0]  # [batch_size, seq_len]
        token_pos_2 = pos_tag_ids[:, :, 1]
        token_pos_3 = pos_tag_ids[:, :, 2]

        pos_embed_1 = self.pos_embedding_1(token_pos_1)
        pos_embed_2 = self.pos_embedding_2(token_pos_2)
        pos_embed_3 = self.pos_embedding_3(token_pos_3)

        # Eojeol
        # [batch_size, max_seq_len, max_eojeol_len]
        eojeol_embed = self.eojeol_embedding(eojeol_ids)

        # Entity
        # entity_embed = self.entity_embedding(entity_ids)
        # eojeol_entity_tensor = self._make_eojeol_entity_embed(eojeol_ids=eojeol_ids, entity_ids=entity_ids)
        # eojeol_entity_embed = self.eojeol_entity_embedding(eojeol_entity_tensor)

        # All Features Concat
        concat_pos_embed = torch.concat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)
        # [64, 128, 1202], [64, 128, 1328]
        concat_all_embed = torch.concat([plm_last_hidden, concat_pos_embed, eojeol_embed], dim=-1)

        # Transformer Encoder
        extend_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extend_attention_mask = extend_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extend_attention_mask = (1.0 - extend_attention_mask) * -10000.0

        t_enc_outputs = self.trans_encoder(concat_all_embed, extend_attention_mask)
        t_enc_outputs = t_enc_outputs[-1]
        t_enc_outputs = self.dropout(t_enc_outputs)

        # Classifier for CRF
        logits = self.ffn_2(t_enc_outputs)

        # CRF
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, reduction="mean"), \
                                               self.crf.decode(logits)
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags