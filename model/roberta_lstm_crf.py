import torch
import torch.nn as nn

from transformers import (
    AutoModel, RobertaPreTrainedModel
)

from model.crf_layer import CRF

#===============================================================
class RoBERTa_LSTM_CRF(RobertaPreTrainedModel):
#===============================================================
    def __init__(self, config):
        super(RoBERTa_LSTM_CRF, self).__init__(config)
        self.max_seq_len = 128
        self.num_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels
        self.pos_embed_out_dim = 100

        # POS Embedding
        self.pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        # PLM
        '''
            @ Note
                AutoModel.from_config()
                Loading a model from its configuration file does not load the model weights. 
                It only affects the model’s configuration. 
                Use from_pretrained() to load the model weights.
        '''
        self.roberta = AutoModel.from_pretrained("klue/roberta-base", config=config)
        self.lstm_dim_size = config.hidden_size + (self.pos_embed_out_dim * 3)
        self.lstm = nn.LSTM(input_size=self.lstm_dim_size,
                            hidden_size=self.lstm_dim_size,
                            num_layers=1, batch_first=True, dropout=config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.lstm_dim_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)


        self.post_init()


    def forward(
            self, input_ids, attention_mask, token_type_ids, pos_tag_ids,
            input_seq_len=None, labels=None, token_seq_len=None, eojeol_ids=None #, entity_ids=None
    ):
        # POS Embedding
        # pos_tag_ids : [batch_size, seq_len, num_pos_tags]
        pos_tag_1 = pos_tag_ids[:, :, 0]  # [batch_size, seq_len]
        pos_tag_2 = pos_tag_ids[:, :, 1]  # [batch_size, seq_len]
        pos_tag_3 = pos_tag_ids[:, :, 2]  # [batch_size, seq_len]

        pos_embed_1 = self.pos_embedding_1(pos_tag_1)  # [batch_size, seq_len, pos_tag_embed]
        pos_embed_2 = self.pos_embedding_2(pos_tag_2)  # [batch_size, seq_len, pos_tag_embed]
        pos_embed_3 = self.pos_embedding_3(pos_tag_3)  # [batch_size, seq_len, pos_tag_embed]

        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        concat_embed = torch.concat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)
        concat_embed = torch.concat([sequence_output, concat_embed], dim=-1)
        lstm_out, _ = self.lstm(concat_embed)  # [batch_size, seq_len, hidden_size]
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)

        # CRF
        if labels is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(
                logits)  # mask=attention_mask.byte())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags

if "__main__" == __name__:
    from transformers import AutoConfig
    a = AutoConfig.from_pretrained("klue/roberta-base")
    print(a)