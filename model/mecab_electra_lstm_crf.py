import torch
import torch.nn as nn

from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from model.crf_layer import CRF

#==============================================================
class ELECTRA_MECAB(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config):
        super(ELECTRA_MECAB, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.num_labels = config.num_labels

        self.pos_embed_out_dim = 128
        self.dropout_rate = 0.3

        '''
            @ Note
                AutoModel.from_config()
                Loading a model from its configuration file does not load the model weights. 
                It only affects the model’s configuration. 
                Use from_pretrained() to load the model weights.
        '''

        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # LSTM
        self.lstm_dim_size = config.hidden_size # + (128 * self.num_ne_pos) + (128 * self.num_josa_pos) # + (self.pos_embed_out_dim * 2)
        self.lstm = nn.LSTM(input_size=self.lstm_dim_size, hidden_size=(self.lstm_dim_size // 2),
                            num_layers=1, batch_first=True, bidirectional=True, dropout=self.dropout_rate)

        # Classifier
        self.classifier = nn.Linear(self.lstm_dim_size, config.num_labels)
        # self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    #===================================
    def forward(self,
                input_ids, token_type_ids, attention_mask,
                labels=None, pos_tag_ids=None, eojeol_ids=None
    ):
    #===================================
        # pos embedding
        # pos_tag_ids : [batch_size, seq_len, num_pos_tags]

        electra_outputs = self.electra(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        electra_outputs = electra_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]
        # concat_embed = torch.concat([electra_outputs, ne_pos_embed, josa_pos_embed], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(electra_outputs) # [batch_size, seq_len, hidden_size]

        # Classifier
        logits = self.classifier(lstm_out) # [128, 128, 31]

        # Get Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

        # crf
        # if labels is not None:
        #     log_likelihood, sequence_of_tags = self.crf(emissions=logits, tags=labels, mask=attention_mask.bool(),
        #                                                 reduction="mean"), self.crf.decode(logits, mask=attention_mask.bool())
        #     return log_likelihood, sequence_of_tags
        # else:
        #     sequence_of_tags = self.crf.decode(logits)
        #     return sequence_of_tags

    #===================================
    def _make_ne_and_josa_pos_tensor(self, src_pos_ids):
    #===================================
        # src_pos_ids : [batch, seq_len, num_pos]
        batch_size, max_seq_len, num_pos = src_pos_ids.size()
        ne_pos_embed = torch.zeros(batch_size, max_seq_len, self.num_ne_pos, dtype=torch.long)
        josa_pos_embed = torch.zeros(batch_size, max_seq_len, self.num_josa_pos, dtype=torch.long)
        '''
            [NNG, NNP, SN, NNB, NR] + Later (SW)
            [JKS, JKC, JKG, JKO, JKB, JKV, JKQ, JX, JC]
        '''

        ne_pos_list = ["NNG", "NNP", "SN", "NNB", "NR"]
        ne_pos_label2id = {label: i for i, label in enumerate(ne_pos_list)}

        josa_list = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"]
        josa_label2id = {label: i for i, label in enumerate(josa_list)}

        for batch_idx, (batch_ne_item, batch_josa_item) in enumerate(zip(ne_pos_embed, josa_pos_embed)):
            for seq_idx in range(max_seq_len):
                ne_pos_one_hot = torch.zeros(self.num_ne_pos, dtype=torch.long)
                josa_one_hot = torch.zeros(self.num_josa_pos, dtype=torch.long)
                curr_seq_pos = src_pos_ids[batch_idx, seq_idx] # [num_pos, ]
                for ne_pos_key, ne_pos_ids in ne_pos_label2id.items():
                    if self.pos_label2id[ne_pos_key] in curr_seq_pos:
                        ne_pos_one_hot[ne_pos_ids] = 1
                for josa_key, josa_ids in josa_label2id.items():
                    if self.pos_label2id[josa_key] in curr_seq_pos:
                        josa_one_hot[josa_ids] = 1
                batch_ne_item[seq_idx] = ne_pos_one_hot
                batch_josa_item[seq_idx] = josa_one_hot

        return ne_pos_embed, josa_pos_embed

    #===================================
    def _gate_network(self, lhs_embed, rhs_embed):
    #===================================
        concat_embed = torch.cat([lhs_embed, rhs_embed], -1)
        context_gate = self.gate_sigmoid(self.gate_layer(concat_embed))
        return torch.add(context_gate * lhs_embed, (1. - context_gate) * rhs_embed)