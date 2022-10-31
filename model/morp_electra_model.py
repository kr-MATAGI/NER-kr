import torch
import torch.nn as nn

from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from model.crf_layer import CRF

#==============================================================
class ELECTRA_MECAB_MORP(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config):
        super(ELECTRA_MECAB_MORP, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.num_labels = config.num_labels

        self.num_ne_pos = config.num_ne_pos
        self.num_josa_pos = config.num_josa_pos
        self.pos_id2label = config.pos_id2label
        self.pos_label2id = config.pos_label2id

        self.pos_embed_out_dim = 128
        self.dropout_rate = 0.3

        '''
            @ Note
                AutoModel.from_config()
                Loading a model from its configuration file does not load the model weights. 
                It only affects the model’s configuration.
                Use from_pretrained() to load the model weights.
        '''
        # self.gate_layer = nn.Linear(config.hidden_size*2, config.hidden_size)
        # self.gate_sigmoid = nn.Sigmoid()

        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # POS tag embedding
        # self.ne_pos_embedding = nn.Embedding(self.num_ne_pos, self.pos_embed_out_dim // 2)
        # self.josa_pos_embedding = nn.Embedding(self.num_josa_pos, self.pos_embed_out_dim)

        # Morp Embedding
        self.morp_embedding = nn.Embedding(self.max_seq_len, self.max_seq_len)

        # LSTM
        # self.lstm_dim_size = config.hidden_size + ((self.pos_embed_out_dim // 2) * self.num_ne_pos) + \
        #                      (self.pos_embed_out_dim * self.num_josa_pos)
        self.lstm_dim_size = config.hidden_size + (self.pos_embed_out_dim * self.num_josa_pos)
        self.lstm = nn.LSTM(input_size=self.lstm_dim_size, hidden_size=(self.lstm_dim_size // 2),
                            num_layers=1, batch_first=True, bidirectional=True)

        # Classifier
        self.classifier = nn.Linear(self.lstm_dim_size, config.num_labels)
        # self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    #===================================
    def forward(self,
                input_ids, token_type_ids, attention_mask,
                labels=None, pos_tag_ids=None,
                morp_ids=None, ne_pos_one_hot=None, josa_pos_one_hot=None,
    ):
    #===================================
        electra_outputs = self.electra(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        electra_outputs = electra_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]

        # Use POS Embedding
        # ne_pos_embed, josa_pos_embed = self._make_ne_and_josa_pos_embedding(ne_one_hot=ne_pos_one_hot,
        #                                                                     josa_one_hot=josa_pos_one_hot)
        josa_pos_embed = self._make_ne_and_josa_pos_embedding(ne_one_hot=ne_pos_one_hot,
                                                              josa_one_hot=josa_pos_one_hot)

        # Make Morp Tokens - [batch_size, seq_len, seq_len]
        # morp_boundary_embed = self._detect_morp_boundary(last_hidden_size=electra_outputs.size(),
        #                                                  device=electra_outputs.device,
        #                                                  morp_ids=morp_ids)
        # morp_tensors = morp_boundary_embed @ electra_outputs

        # Concat
        concat_embed = torch.concat([electra_outputs, josa_pos_embed], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(concat_embed) # [batch_size, seq_len, hidden_size]

        # Classifier
        logits = self.classifier(lstm_out) # [128, 128, 31]

        # Get LossE
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
    def _make_ne_and_josa_pos_embedding(
            self,
            ne_one_hot: torch.Tensor,
            josa_one_hot: torch.Tensor,
    ):
    #===================================
        # ne_pos_embedding = self.ne_pos_embedding(ne_one_hot) # [batch, seq_len, num_ne_pos, dim_outputs]
        josa_pos_embedding = self.josa_pos_embedding(josa_one_hot) # [batch, seq_len, num_ne_pos, dim_outputs]

        '''
            ne_pos_embedding.shape: [batch_size, seq_len, 320]
            josa_pos_embedding.shape: [batch_size, seq_len, 1152]
        '''
        # ne_pos_embedding = ne_pos_embedding.view([ne_pos_embedding.shape[0], ne_pos_embedding.shape[1], -1])
        josa_pos_embedding = josa_pos_embedding.view([josa_pos_embedding.shape[0], josa_pos_embedding.shape[1], -1])

        # return ne_pos_embedding, josa_pos_embedding
        return josa_pos_embedding


    #===================================
    def _detect_morp_boundary(
            self,
            last_hidden_size: torch.Size,
            device: str,
            morp_ids: torch.Tensor
    ):
    #===================================
        batch_size, seq_len, hidden_dim = last_hidden_size
        new_morp_tensors = torch.zeros(batch_size, seq_len, dtype=torch.long)

        for batch_idx in range(batch_size):
            cur_idx = 0
            morp_tensor = torch.zeros(seq_len, dtype=torch.long)
            for mp_idx, mp_tok_cnt in enumerate(morp_ids[batch_idx]):
                tok_cnt = mp_tok_cnt.detach().cpu().item()
                if 0 == tok_cnt:
                    break
                for _ in range(tok_cnt):
                    if seq_len <= cur_idx:
                        break
                    morp_tensor[cur_idx] = mp_idx
                    cur_idx += 1
            new_morp_tensors[batch_idx] = morp_tensor

        new_morp_tensors = new_morp_tensors.to(device)
        morp_boundary_embed = self.morp_embedding(new_morp_tensors)

        return morp_boundary_embed

    #===================================
    def _gate_network(self, lhs_embed, rhs_embed):
    #===================================
        concat_embed = torch.cat([lhs_embed, rhs_embed], -1)
        context_gate = self.gate_sigmoid(self.gate_layer(concat_embed))
        return torch.add(context_gate * lhs_embed, (1. - context_gate) * rhs_embed)