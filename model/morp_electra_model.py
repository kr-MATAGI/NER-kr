import torch
import torch.nn as nn
import math

from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from model.crf_layer import CRF
from model.char_cnn import CharCNN

''' CharELMo '''
import pickle
from model.charELMo import CharELMo

#==============================================================
class AttentionConfig:
#==============================================================
    def __init__(self,
                 num_heads: int = 12,
                 hidden_size: int = 768,
                 dropout_prob: float = 0.1,
                 ):
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

#==============================================================
class Attention(nn.Module):
#==============================================================
    def __init__(self, config: AttentionConfig):
        super(Attention, self).__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # attention_scores : [64, 8, 25, 25]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.o_proj(context_layer)

        return attention_output

#==============================================================
class ELECTRA_MECAB_MORP(ElectraPreTrainedModel):
#==============================================================
    def __init__(self, config):
        super(ELECTRA_MECAB_MORP, self).__init__(config)
        self.max_seq_len = config.max_seq_len
        self.num_labels = config.num_labels
        self.dropout_rate = 0.1

        self.num_ne_pos = config.num_ne_pos
        self.num_josa_pos = config.num_josa_pos
        self.pos_id2label = config.pos_id2label
        self.pos_label2id = config.pos_label2id

        # POS
        self.pos_embed_out_dim = 128

        # Morp
        # self.morp_embed_out_dim = 128

        # Char-level
        self.elmo_vocab_size = config.elmo_vocab_size

        '''
            @ Note
                AutoModel.from_config()
                Loading a model from its configuration file does not load the model weights. 
                It only affects the model’s configuration.
                Use from_pretrained() to load the model weights.
        '''
        # self.gate_layer = nn.Linear(config.hidden_size*2, config.hidden_size)
        # self.gate_sigmoid = nn.Sigmoid()

        # ELECTRA
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
        self.dropout = nn.Dropout(self.dropout_rate)

        # CharELMo
        #self.charELMo = CharELMo(vocab_size=self.elmo_vocab_size, mode="repr")
        #self.charELMo.load_state_dict(torch.load("C:/Users/MATAGI/Desktop/Git/NER_Private/model/charELMo/16_model.pth"))

        # Char-level Embedding
        '''
        self.max_cnn_input = 30 # CNN에 입력으로 최대 몇 글자가 들어갈지 (글자 * 3(초/중/종성))
        self.char_cnn = CharCNN(vocab_size=self.char_vocab_size,
                                seq_len=self.max_seq_len)
        '''

        # POS tag embedding
        # self.ne_pos_embedding = nn.Embedding(self.num_ne_pos, self.pos_embed_out_dim // 2)
        # self.josa_pos_embedding = nn.Embedding(self.num_josa_pos, self.pos_embed_out_dim)

        # Morp Embedding
        # self.morp_embedding = nn.Embedding(self.max_seq_len, self.max_seq_len)

        # LSTM Encoder
        # self.lstm_dim_size = config.hidden_size + ((self.pos_embed_out_dim // 2) * self.num_ne_pos) + \
        #                      (self.pos_embed_out_dim * self.num_josa_pos)
        self.lstm_dim = config.hidden_size #* 3
        self.encoder = nn.LSTM(input_size=self.lstm_dim, hidden_size=(config.hidden_size // 2),
                               num_layers=1, batch_first=True, bidirectional=True)

        # Attention
        # self.attn_hidden_dim = config.hidden_size
        # self.attn_config = AttentionConfig(hidden_size=self.attn_hidden_dim)
        # self.attn_layer = Attention(self.attn_config)

        # Classifier
        self.classifier_dim = config.hidden_size
        self.classifier = nn.Linear(self.classifier_dim, config.num_labels)
        # self.crf = CRF(num_tags=config.num_labels, batch_first=True)

        # Initialize weights and apply final processing
        self.post_init()

    #===================================
    def forward(self,
                input_ids, token_type_ids, attention_mask,
                label_ids=None, pos_tag_ids=None,
                morp_ids=None, ne_pos_one_hot=None, josa_pos_one_hot=None,
                jamo_ids=None, jamo_boundary=None, sents=None
                ):
    #===================================
        '''
            char_ids: 초/중/종성, 기호, 숫자, 영어의 vocab_ids
            char_boundary: 각 형태소별 문자 길이
        '''

        electra_outputs = self.electra(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        electra_outputs = electra_outputs.last_hidden_state # [batch_size, seq_len, hidden_size]

        # Use POS Embedding
        # ne_pos_embed, josa_pos_embed = self._make_ne_and_josa_pos_embedding(ne_one_hot=ne_pos_one_hot,
        #                                                                     josa_one_hot=josa_pos_one_hot)
        # josa_pos_embed = self._make_ne_and_josa_pos_embedding(ne_one_hot=ne_pos_one_hot,
        #                                                       josa_one_hot=josa_pos_one_hot)

        ''' Make Morp Tokens - [batch_size, seq_len, seq_len] '''
        # morp_boundary_embed = self._detect_morp_boundary(last_hidden_size=electra_outputs.size(),
        #                                                  device=electra_outputs.device,
        #                                                  morp_ids=morp_ids)
        # morp_embed = morp_boundary_embed @ electra_outputs

        ''' Make Char-Level Embedding '''
        '''
        device = electra_outputs.device
        tensor_size = electra_outputs.size() # [batch, seq_len, hidden]
        '''
            #jamo_ids.shape: [batch_size, seq_len, 3]
            #jamo_boundary: [batch_size, seq_len]
        '''
        # [batch_size, vocab_size, seq_len * 3]
        char_lvl_tensor = self._make_jamo_tensor(char_ids=jamo_ids, char_boundary=jamo_boundary,
                                                 device=device, tensor_size=tensor_size)
        new_char_lvl_tensor = None
        for batch_idx in range(tensor_size[0]):
            boundary = jamo_boundary[batch_idx]
            batch_char_tensor = char_lvl_tensor[batch_idx]
            start_bdry = 0
            new_seq_tensor = None
            for bdry in boundary:
                extract_tensor = batch_char_tensor[:, start_bdry:start_bdry+bdry.item()]
                if self.max_cnn_input > extract_tensor.shape[1]:
                    diff_size = self.max_cnn_input - extract_tensor.shape[1]
                    empty_pad_tensor = torch.zeros(self.char_vocab_size, diff_size, device=torch.device(device))
                    extract_tensor = torch.hstack([extract_tensor, empty_pad_tensor])
                extract_tensor = extract_tensor.unsqueeze(0) # [1, vocab_size, 형태소 최대 길이]
                start_bdry += bdry.item()
                cnn_out = self.char_cnn(extract_tensor)  # [batch, last_linear_embed]
                cnn_out = cnn_out
                if new_seq_tensor is None:
                    new_seq_tensor = cnn_out
                else:
                    new_seq_tensor = torch.vstack([new_seq_tensor, cnn_out])
            if new_char_lvl_tensor is None:
                new_char_lvl_tensor = new_seq_tensor.unsqueeze(0)
            else:
                new_char_lvl_tensor = torch.vstack([new_char_lvl_tensor, new_seq_tensor.unsqueeze(0)])
        new_char_lvl_tensor = new_char_lvl_tensor
        '''

        # CharELMo
        '''
            elmo_x : [batch_size, seq_len, elmo_embed_dim]
            elmo_layer_1_out : [batch_size, seq_len, elmo_lstm_hidden * 2]
            elmo_layer_2_out : [batch_size, seq_len, elmo_lstm_hidden * 2]
            elmo_repr : [batch_size, seq_len, elmo_lstm_hidden * 2]
        '''
        #elmo_x, elmo_layer_1_out, elmo_layer_2_out = self.charELMo(sents)
        ''' elmo_x 써야 되는가 차원 안맞긴 한데 안써도 된다고 본거 같긴함 (optional)'''

        # Concat
        #concat_embed = torch.concat([electra_outputs, elmo_layer_2_out], dim=-1)

        # LSTM
        '''
            output: [seq_len, batch_size, hidden_dim * n_direction]
            hidden: [n_layers * n_directions, batch_size, hidden_dim]
            cell: [n_layers * n_directions, batch_size, hidden_dim]
        '''
        enc_out, (enc_h_n, enc_c_n) = self.encoder(electra_outputs) # [batch_size, seq_len, hidden_size]

        # Attention
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        # attn_out = self.attn_layer(enc_out, attention_mask)

        # Classifier
        logits = self.classifier(enc_out)

        # Get LossE
        loss = None
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), label_ids.view(-1))

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
    def _make_jamo_tensor(self,
                          char_ids: torch.Tensor,
                          char_boundary: torch.Tensor,
                          device, tensor_size
                          ):
    #===================================
        '''
            char_ids: [batch_size, seq_len, 3]
            char_boundary: [batch_size, seq_len]
            tensor_size : [batch, seq_len, hidden]
            return:
                stacked_tensors: [batch, seq_len, vocab_size's one_hot]
        '''

        stacked_tensors = torch.zeros(tensor_size[0], tensor_size[1] * 3, self.char_vocab_size,
                                      device=torch.device(device))
        for batch_idx in range(tensor_size[0]):
            batch_char_ids = char_ids[batch_idx]

            seq_tensor = None
            for seq_ch in batch_char_ids:
                first_one_hot = torch.zeros(self.char_vocab_size, device=torch.device(device))
                second_one_hot = torch.zeros(self.char_vocab_size, device=torch.device(device))
                third_one_hot = torch.zeros(self.char_vocab_size, device=torch.device(device))

                first_one_hot[seq_ch[0]] = 1 # [1, vocab_size]
                second_one_hot[seq_ch[1]] = 1
                third_one_hot[seq_ch[2]] = 1

                if seq_tensor is None:
                    seq_tensor = torch.vstack([first_one_hot, second_one_hot, third_one_hot])
                else:
                    seq_tensor = torch.vstack([seq_tensor, first_one_hot, second_one_hot, third_one_hot])
            stacked_tensors[batch_idx] = seq_tensor
        stacked_shape = stacked_tensors.shape
        # [batch_size, vocab_size, seq_len * 3]
        stacked_tensors = stacked_tensors.reshape(stacked_shape[0], stacked_shape[2], stacked_shape[1])
        return stacked_tensors

    #===================================
    def _gate_network(self, lhs_embed, rhs_embed):
    #===================================
        concat_embed = torch.cat([lhs_embed, rhs_embed], -1)
        context_gate = self.gate_sigmoid(self.gate_layer(concat_embed))
        return torch.add(context_gate * lhs_embed, (1. - context_gate) * rhs_embed)