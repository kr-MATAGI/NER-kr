import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from model.transformer_encoder import Trans_Encoder, Enc_Config
from model.classifier.span_classifier import MultiNonLinearClassifier

from model.crf_layer import CRF

#==============================================================
class CHAR_ELECTRA_POS_LSTM(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config):
        super(CHAR_ELECTRA_POS_LSTM, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.num_labels = config.num_labels
        self.num_pos_labels = config.num_pos_labels
        self.pos_embed_dim = 100
        self.dropout_rate = 0.1
        self.num_flag_pos = 15 # "##" 대신 마지막 인덱스에 1

        ''' POS Embedding '''
        self.pos_embedding_1 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_2 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)
        self.pos_embedding_3 = nn.Embedding(self.num_pos_labels, self.pos_embed_out_dim)

        ''' POS Bit Flag'''
        # self.pos_flag_embedding = nn.Embedding(self.num_flag_pos, self.pos_embed_dim)

        '''
            @ Note
                AutoModel.from_config()
                Loading a model from its configuration file does not load the model weights. 
                It only affects the model’s configuration. 
                Use from_pretrained() to load the model weights.
        '''
        self.electra = ElectraModel.from_pretrained("monologg/kocharelectra-base-discriminator", config=config)

        # LSTM
        self.lstm_dim_size = config.hidden_size + (self.pos_embed_dim * 3)
        self.lstm = nn.LSTM(input_size=self.lstm_dim_size, hidden_size=(self.lstm_dim_size // 2),
                            num_layers=1, batch_first=True, bidirectional=True, dropout=self.dropout_rate)

        # Classifier
        self.classifier = nn.Linear(self.lstm_dim_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        ''' 뒷 부분에서 POS Embedding 추가하는 거 '''
        # self.post_pos_embed_dim = config.hidden_size + (self.pos_embed_dim * self.num_flag_pos)
        # self.post_pos_embedding = MultiNonLinearClassifier(self.post_pos_embed_dim,
        #                                                    config.num_labels, self.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

    #===================================
    def forward(self,
                input_ids, token_type_ids, attention_mask,
                label_ids=None, pos_ids=None
    ):
    #===================================
        # pos embedding
        # pos_tag_ids : [batch_size, seq_len, num_pos_tags]
        pos_embed_1 = self.pos_embedding_1(pos_ids[:, :, 0]) # [batch_size, seq_len, pos_tag_embed]
        pos_embed_2 = self.pos_embedding_2(pos_ids[:, :, 1])  # [batch_size, seq_len, pos_tag_embed]
        pos_embed_3 = self.pos_embedding_3(pos_ids[:, :, 2])  # [batch_size, seq_len, pos_tag_embed]

        pos_embed_1 = F.relu(pos_embed_1)
        pos_embed_2 = F.relu(pos_embed_2)
        pos_embed_3 = F.relu(pos_embed_3)

        electra_outputs = self.electra(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        electra_outputs = electra_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]

        ''' Add POS Embedding '''
        add_pos_embed = torch.add(pos_embed_1, pos_embed_2)
        add_pos_embed = torch.add(add_pos_embed, pos_embed_3)
        concat_pos = torch.concat([electra_outputs, add_pos_embed], dim=-1)

        # POS Flag
        '''
        pos_flag_out = self.pos_flag_embedding(pos_ids)  # [batch, seq_len, num_pos, pos_emb_dim]
        pos_flag_out = F.relu(pos_flag_out)
        pos_flag_size = pos_flag_out.size()
        pos_flag_out = pos_flag_out.reshape(pos_flag_size[0], pos_flag_size[1], -1)
        '''

        # concat_pos_embed = torch.concat([pos_embed_1, pos_embed_2, pos_embed_3], dim=-1)
        # concat_embed = torch.concat([electra_outputs, pos_flag_out], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(concat_pos) # [batch_size, seq_len, hidden_size]

        # Classifier
        logits = self.classifier(lstm_out) # [128, 128, 31]

        ''' 뒷 부분에서 POS Embedding 추가하는 거 '''
        # concat_pos_flag = torch.cat([lstm_out, pos_flag_out], dim=-1)
        # logits = self.post_pos_embedding(concat_pos_flag)

        # Get Loss
        # loss = None
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
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