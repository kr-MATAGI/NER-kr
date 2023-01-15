import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from model.classifier.span_classifier import MultiNonLinearClassifier
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
        self.num_flag_pos = 11
        self.pos_embed_dim = 100

        ''' POS Embedding '''
        # self.pos_embedding_1 = nn.Embedding(self.num_pos, self.pos_embed_dim)
        # self.pos_embedding_2 = nn.Embedding(self.num_pos, self.pos_embed_dim)
        # self.pos_embedding_3 = nn.Embedding(self.num_pos, self.pos_embed_dim)
        # self.pos_embedding_4 = nn.Embedding(self.num_pos, self.pos_embed_dim)

        # POS Flag
        ''' POS bit flag 추가하는 거 '''
        self.pos_flag_embedding = nn.Embedding(self.num_flag_pos, self.pos_embed_dim)

        # ELECTRA
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)

        # LSTM
        self.lstm_dim = config.hidden_size + (self.pos_embed_dim * self.num_flag_pos)
        self.encoder = nn.LSTM(input_size=self.lstm_dim, hidden_size=(self.lstm_dim // 2),
                               num_layers=1, batch_first=True, bidirectional=True)

        ''' 뒷 부분에서 POS bit flag 추가하는 거 '''
        # self.post_pos_embed_dim = config.hidden_size + (self.pos_embed_dim * self.num_flag_pos)
        # self.post_pos_embedding = MultiNonLinearClassifier(self.post_pos_embed_dim,
        #                                                    config.num_labels, self.dropout_rate)

        # Classifier
        ''' 앞 부분에서 POS 추가할 때 사용 '''
        self.classifier_dim = config.hidden_size + (self.pos_embed_dim * self.num_flag_pos)
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

        # POS
        # pos_out_1 = self.pos_embedding_1(pos_ids[:, :, 0])
        # pos_out_2 = self.pos_embedding_2(pos_ids[:, :, 1])
        # pos_out_3 = self.pos_embedding_3(pos_ids[:, :, 2])
        # pos_out_4 = self.pos_embedding_4(pos_ids[:, :, 3])

        # pos_out_1 = F.relu(pos_out_1)
        # pos_out_2 = F.relu(pos_out_2)
        # pos_out_3 = F.relu(pos_out_3)
        # pos_out_4 = F.relu(pos_out_4)
        # concat_pos = torch.concat([pos_out_1, pos_out_2, pos_out_3], dim=-1)

        ''' Add POS Embedding '''
        # add_pos_embed = torch.add(pos_out_1, pos_out_2)
        # add_pos_embed = torch.add(add_pos_embed, pos_out_3)
        # concat_pos = torch.concat([electra_outputs, add_pos_embed], dim=-1)

        ''' POS Flag '''
        pos_flag_out = self.pos_flag_embedding(pos_ids)  # [batch, seq_len, num_pos, pos_emb_dim]
        pos_flag_out = F.relu(pos_flag_out)
        pos_flag_size = pos_flag_out.size()
        pos_flag_out = pos_flag_out.reshape(pos_flag_size[0], pos_flag_size[1], -1)

        # LSTM
        concat_pos = torch.concat([electra_outputs, pos_flag_out], dim=-1)
        enc_out, _ = self.encoder(concat_pos) # [batch_size, seq_len, hidden_size]
        # Classifier
        # concat_pos = torch.concat([enc_out, add_pos_embed], dim=-1)
        logits = self.classifier(enc_out)

        ''' 뒷 부분에서 POS Embedding 추가하는 거 '''
        # concat_pos_flag = torch.cat([enc_out, concat_pos], dim=-1)
        # logits = self.post_pos_embedding(concat_pos_flag)

        # Get LossE
        # loss = None
        # if label_ids is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.config.num_labels), label_ids.view(-1))
        #
        # return TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits
        # )

        # crf
        if label_ids is not None:
            log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=label_ids, mask=attention_mask.bool(),
                                                        reduction="mean"), self.crf.decode(logits, mask=attention_mask.bool())
            return log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(logits)
            return sequence_of_tags
