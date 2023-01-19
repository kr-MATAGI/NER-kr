import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, ElectraPreTrainedModel

from model.crf_layer import CRF

#==============================================================
class ELECTRA_MECAB_MORP(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config):
        super(ELECTRA_MECAB_MORP, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.num_labels = config.num_labels
        self.num_pos_tag = config.num_pos_labels
        self.dropout_rate = 0.1
        self.num_flag_pos = 12
        self.pos_embed_dim = 128

        # POS Flag
        ''' POS bit flag 추가하는 거 '''
        self.pos_flag_embedding = nn.Embedding(self.num_flag_pos, self.pos_embed_dim)

        # ELECTRA
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # LSTM - Encoder
        self.lstm_dim = config.hidden_size + (self.pos_embed_dim * self.num_flag_pos)
        self.encoder = nn.LSTM(input_size=self.lstm_dim, hidden_size=(self.lstm_dim // 2),
                               num_layers=1, batch_first=True, bidirectional=True)

        self.mt_attn = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True)

        # Classifier
        self.classifier_dim = config.hidden_size + (self.pos_embed_dim * self.num_flag_pos) + config.hidden_size
        self.classifier = nn.Linear(self.classifier_dim, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()


    #===================================
    def forward(self,
                input_ids, token_type_ids, attention_mask,
                label_ids=None, pos_ids=None
                ):
    #===================================
        electra_outputs = self.electra(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        electra_outputs = electra_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]
        electra_outputs = self.dropout(electra_outputs)

        ''' POS Flag '''
        pos_flag_out = self.pos_flag_embedding(pos_ids)  # [batch, seq_len, num_pos, pos_emb_dim]
        pos_flag_out = F.relu(pos_flag_out)
        pos_flag_size = pos_flag_out.size()
        pos_flag_out = pos_flag_out.reshape(pos_flag_size[0], pos_flag_size[1], -1)

        # LSTM
        concat_pos = torch.concat([electra_outputs, pos_flag_out], dim=-1)
        enc_out, hidden = self.encoder(concat_pos) # [batch_size, seq_len, hidden_size]
        # hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        attn_output, attn_output_weights = self.mt_attn(query=electra_outputs, key=electra_outputs, value=electra_outputs)
        enc_out = torch.concat([enc_out, attn_output], dim=-1)

        # Classifier
        logits = self.classifier(enc_out)

        # crf
        if label_ids is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=label_ids, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(logits, mask=attention_mask.bool())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags