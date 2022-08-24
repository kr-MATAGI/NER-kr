import torch
import torch.nn as nn

from transformers import ElectraModel, ElectraPreTrainedModel
from model.crf_layer import CRF

#==============================================================
class ELECTRA_POS_LSTM(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config):
        super(ELECTRA_POS_LSTM, self).__init__(config)
        self.max_seq_len = 128 #config.max_seq_len
        self.max_eojeol_len = 48

        self.num_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels
        self.pos_embed_out_dim = 100
        self.entity_embed_out_dim = 128

        self.dropout_rate = 0.1

        # pos tag embedding
        self.pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        # bert + lstm
        '''
            @ Note
                AutoModel.from_config()
                Loading a model from its configuration file does not load the model weights. 
                It only affects the model’s configuration. 
                Use from_pretrained() to load the model weights.
        '''
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)

        # eojeol
        # self.eojeol_embedding = nn.Embedding(self.max_seq_len, self.max_eojeol_len)

        # entity
        # self.entity_embedding = nn.Embedding(self.max_seq_len, self.entity_embed_out_dim)

        self.lstm_dim_size = config.hidden_size + (self.pos_embed_out_dim * 3)# + self.max_eojeol_len# + self.entity_embed_out_dim
        self.lstm = nn.LSTM(input_size=self.lstm_dim_size, hidden_size=self.lstm_dim_size,
                            num_layers=1, batch_first=True, dropout=self.dropout_rate)
        self.classifier = nn.Linear(self.lstm_dim_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        self.post_init()

    #===================================
    def forward(self,
                input_ids, token_type_ids, attention_mask,
                token_seq_len=None, labels=None, pos_tag_ids=None,
                eojeol_ids=None, entity_ids=None
    ):
    #===================================
        # pos embedding
        # pos_tag_ids : [batch_size, seq_len, num_pos_tags]
        pos_tag_1 = pos_tag_ids[:, :, 0] # [batch_size, seq_len]
        pos_tag_2 = pos_tag_ids[:, :, 1] # [batch_size, seq_len]
        pos_tag_3 = pos_tag_ids[:, :, 2] # [batch_size, seq_len]

        pos_embed_1 = self.pos_embedding_1(pos_tag_1) # [batch_size, seq_len, pos_tag_embed]
        pos_embed_2 = self.pos_embedding_2(pos_tag_2)  # [batch_size, seq_len, pos_tag_embed]
        pos_embed_3 = self.pos_embedding_3(pos_tag_3)  # [batch_size, seq_len, pos_tag_embed]

        # eojeol
        # eojeol_embed = self.eojeol_embedding(eojeol_ids)

        # entity
        # entity_embed = self.entity_embedding(entity_ids)

        outputs = self.electra(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        sequence_output = outputs.last_hidden_state # [batch_size, seq_len, hidden_size]

        concat_pos_embed = torch.concat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)
        concat_embed = torch.concat([sequence_output, concat_pos_embed], dim=-1)
        # concat_embed = torch.concat([concat_embed, eojeol_embed, entity_embed], dim=-1)
        #concat_embed = torch.concat([concat_embed, eojeol_embed], dim=-1)
        lstm_out, _ = self.lstm(concat_embed) # [batch_size, seq_len, hidden_size]
        logits = self.classifier(lstm_out) # [128, 128, 31]

        # crf
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(logits, mask=attention_mask.bool())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags