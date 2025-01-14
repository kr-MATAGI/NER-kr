import copy
import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraPreTrainedModel

from model.classifier.span_classifier import SingleLinearClassifier, MultiNonLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F
from utils.tag_def import MECAB_POS_TAG

#=======================================================
class ElectraSpanNER(ElectraPreTrainedModel):
#=======================================================
    def __init__(self, config):
        super(ElectraSpanNER, self).__init__(config)

        # Init
        self.hidden_size = config.hidden_size
        self.n_class = config.num_labels
        self.ids2label = config.id2label

        self.span_combi_mode = "x,y"
        self.token_len_emb_dim = 50
        self.max_span_width = 8
        self.max_seq_len = 128

        self.span_len_emb_dim = 100
        ''' morp는 origin에서 {'isupper': 1, 'islower': 2, 'istitle': 3, 'isdigit': 4, 'other': 5}'''
        self.pos_emb_dim = 100
        self.n_pos = 14 # 일반/교유 명사 통합
        # self.n_pos = 43 # 모든 품사 사용

        ''' 원본 Git에서는 Method 적용 개수에 따라 달라짐 '''
        self.input_dim = self.hidden_size * 2 + self.token_len_emb_dim + self.span_len_emb_dim #+ (self.pos_emb_dim * self.n_pos)
        self.model_dropout = 0.1

        # loss and softmax
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none') # 이거 none이여야 compute loss 동작
        self.softmax = torch.nn.Softmax(dim=-1) # for predict

        print("self.max_span_width: ", self.max_span_width)
        print("self.tokenLen_emb_dim: ", self.token_len_emb_dim)
        print("self.pos_emb_dim.dim: ", self.pos_emb_dim)
        print("self.input_dim: ", self.input_dim)

        # Model
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)

        self.endpoint_span_extractor = EndpointSpanExtractor(input_dim=self.hidden_size,
                                                             combination=self.span_combi_mode,
                                                             num_width_embeddings=self.max_span_width,
                                                             span_width_embedding_dim=self.token_len_emb_dim,
                                                             bucket_widths=True)

        self.span_embedding = MultiNonLinearClassifier(self.input_dim,
                                                       self.n_class,
                                                       self.model_dropout)

        self.span_len_embedding = nn.Embedding(self.max_span_width + 1, self.span_len_emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.n_pos, self.pos_emb_dim)

    #==============================================
    def forward(self,
                all_span_lens, all_span_idxs_ltoken, real_span_mask_ltoken, input_ids, pos_ids,
                token_type_ids=None, attention_mask=None, span_only_label=None, mode:str = "train",
                label_ids=None
                ):
    #==============================================
        """
        Args:
            loadall: [tokens, token_type_ids, all_span_idxs_ltoken,
                     morph_idxs, span_label_ltoken, all_span_lens, all_span_weights,
                     real_span_mask_ltoken, words, all_span_word, all_span_idxs]
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
            all_span_idxs: the span-idxs on token-level. (bs, n_span)
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        electra_outputs = self.electra(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)

        electra_outputs = electra_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # [batch, n_span, input_dim] : [64, 502, 1586]
        '''
            all_span_rep : [batch, n_span, output_dim]
            all_span_idxs_ltoken : [batch, n_span, 2]
        '''
        all_span_rep = self.endpoint_span_extractor(electra_outputs, all_span_idxs_ltoken.long())

        ''' use span len, not use morp'''
        # n_span : span 개수
        span_len_rep = self.span_len_embedding(all_span_lens) # [batch, n_span, len_dim]
        span_len_rep = F.relu(span_len_rep) # [64, 502, 100]

        # [batch, n_span, num_pos]
        ''' Flag POS Embedding '''
        # span_morp_rep = self.pos_embedding(pos_ids) # [batch, n_span, num_pos, pos_emb_dim]
        # span_morp_rep = F.relu(span_morp_rep)
        # morp_rep_size = span_morp_rep.size()
        # span_morp_rep = span_morp_rep.reshape(morp_rep_size[0], morp_rep_size[1], -1)

        all_span_rep = torch.cat((all_span_rep, span_len_rep), dim=-1)
        # all_span_rep = torch.cat((all_span_rep, span_len_rep, span_morp_rep), dim=-1)
        all_span_rep = self.span_embedding(all_span_rep) # [batch, n_span, n_class] : [64, 502, 16]
        predict_prob = self.softmax(all_span_rep)

        if "eval" == mode:
            loss = self.compute_loss(all_span_rep, span_only_label, real_span_mask_ltoken)
            preds = self.get_predict(predicts=predict_prob, all_span_idxs=all_span_idxs_ltoken)
            return loss, preds
        else:
            loss = self.compute_loss(all_span_rep, span_only_label, real_span_mask_ltoken)
            return loss

    #==============================================
    def compute_loss(self, all_span_rep, span_label_ltoken, real_span_mask_ltoken):
    #==============================================
        '''
        :param all_span_rep:
        :param span_label_ltoken:
        :param real_span_mask_ltoken: attetnion_mask랑 같은 역할하는 것으로 보임 (원래는 IntTensor 일듯)
        :param mode:
        :return:
        '''

        # print("COMPUTE_LOSS::::")
        # print("ALL_SPAN_REP: ", all_span_rep.shape) # [64, 502, 16]
        # print("SPAN_LABEL_LTOKEN: ", span_label_ltoken.shape) # [64, 502, 2]
        # print("REAL_SPAN_MASK: ", real_span_mask_ltoken.shape) # [64, 502]

        batch_size, n_span = span_label_ltoken.size()
        loss = self.cross_entropy(all_span_rep.view(-1, self.n_class),
                                  span_label_ltoken.view(-1))
        loss = loss.view(batch_size, n_span)
        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())
        loss = torch.mean(loss)

        return loss

    #==============================================
    def get_predict(self, predicts, all_span_idxs):
    #==============================================
        '''
            Decode 함수
            predicts = [batch, max_span, num_labels]
        '''
        predicts_max = torch.max(predicts, dim=-1)
        pred_label_max_prob = predicts_max[0] # 스팬별 가질 수 있는 Label prob [batch, n_span]
        pred_label_max_label = predicts_max[1] # 스팬별 가질 수 있는 Label [batch, n_span]

        batch_size, max_span_len, _ = all_span_idxs.size()
        decoded_batches = []
        for batch_idx in range(batch_size):
            batch_pred_span = [] # [span_pair, label]
            check_use_idx = [False for _ in range(self.max_seq_len)]

            # 배치 마다 Span Pair 만들기 (span, 확률, label) -> List[Span Pair]
            # 확률이 높은 span 우선
            span_pair_list = []
            for span_idxs, pred_prob, pred_label in zip(all_span_idxs[batch_idx],
                                                        pred_label_max_prob[batch_idx], pred_label_max_label[batch_idx]):
                span_pair_list.append((span_idxs.tolist(), pred_prob.item(), pred_label.item()))
            span_pair_list = sorted(span_pair_list, key=lambda x: x[1], reverse=True)

            for span_pair in span_pair_list:
                if 0 == span_pair[-1]:
                    continue
                is_break = False
                curr_idxs = [i for i in range(span_pair[0][0], span_pair[0][1] + 1)]
                for c_idx in curr_idxs:
                    if check_use_idx[c_idx]:
                        is_break = True
                        break
                if is_break:
                    continue
                else:
                    batch_pred_span.append((span_pair[0], span_pair[-1]))
                    for s_idx in range(span_pair[0][0], span_pair[0][1] + 1):
                        check_use_idx[s_idx] = True

            decoded_pred = [self.ids2label[0] for _ in range(self.max_seq_len)]
            for span, label in batch_pred_span:
                # print(label, self.ids2label[label])
                s_idx = span[0]
                e_idx = span[1] + 1
                for dec_idx in range(s_idx, e_idx):
                    if dec_idx == s_idx:
                        decoded_pred[dec_idx] = "B-" + self.ids2label[label]
                    else:
                        decoded_pred[dec_idx] = "I-" + self.ids2label[label]
            decoded_batches.append(decoded_pred)
        # end loop, batch
        return decoded_batches

    #==============================================
    def make_pos_embedding(self, pos_ids, all_span_idx_list):
    #==============================================
        batch_size, n_span, _ = all_span_idx_list.size()
        device = all_span_idx_list.device

        mecab_tag2ids = {v: k for k, v in MECAB_POS_TAG.items()} # origin, 1: "NNG"
        target_tag_list = [ # NN은 NNG/NNP 통합
            "NNG", "NNP", "SN", "NNB", "NR", "NNBC",
            "JKS", "JKC", "JKG", "JKO", "JKB", "JX", "JC", "JKV", "JKQ",
        ]

        # target_tag_list = [t for t in mecab_tag2ids.keys() if "O" != t]

        ''' 해당 되는 것의 pos_ids의 새로운 idx '''
        # {1: 0, 2: 1, 43: 2, 3: 3, 5: 4, 4: 5, 16: 6, 17: 7, 18: 8, 19: 9, 20: 10, 23: 11, 24: 12, 21: 13, 22: 14}
        target_tag2ids = {mecab_tag2ids[x]: i for i, x in enumerate(target_tag_list)}
        batch_pos_onehot = torch.zeros((batch_size, n_span, self.n_pos), device=device, dtype=torch.long)
        for batch_idx in range(batch_size):
            curr_pos_ids = pos_ids[batch_idx] # [seq_len, mecab_pos]
            span_idx = 0

            for start_index in range(self.max_seq_len):
                last_end_index = min(start_index + self.max_span_width, self.max_seq_len)
                first_end_index = min(start_index, self.max_seq_len)
                for end_index in range(first_end_index, last_end_index):
                    span_pos = torch.zeros(self.n_pos, device=device, dtype=torch.long)
                    filter_pos = torch.where(1 == curr_pos_ids[start_index:end_index])
                    for f_p in filter_pos[0]:
                        if f_p.item() in target_tag2ids.keys():
                            if 1 == f_p.item() or 0 == f_p.item():
                                span_pos[0] = 1
                            else:
                                span_pos[target_tag2ids[f_p.item()] - 1] = 1
                    batch_pos_onehot[batch_idx, span_idx] = span_pos
                    span_idx += 1
                    if n_span <= span_idx:
                        break
                if n_span <= span_idx:
                    break

        return batch_pos_onehot

    #==============================================
    def make_nn_pos_embedding(self, pos_ids, all_span_idx_list):
    #==============================================
        batch_size, n_span, _ = all_span_idx_list.size()
        device = all_span_idx_list.device

        mecab_tag2ids = {v: k for k, v in MECAB_POS_TAG.items()}  # origin, 1: "NNG"
        target_tag_list = [t for t in mecab_tag2ids.keys() if "O" != t]

        ''' 사용할 POS의 개수가 3개 이므로 '''
        batch_pos_ids = torch.zeros((batch_size, n_span, 3), device=device, dtype=torch.long)
        for batch_idx in range(batch_size):
            curr_pos_ids = pos_ids[batch_idx]
            curr_span_pos_ids = torch.zeros((n_span, 3), device=device, dtype=torch.long)
            for start_index in range(self.max_seq_len):
                last_end_index = min(start_index + self.max_span_width, self.max_seq_len)
                first_end_index = min(start_index, self.max_seq_len)

                for end_index in range(first_end_index, last_end_index):
                    add_pos_cnt = 0
                    add_pos_ids = torch.zeros(3, device=device, dtype=torch.long)
                    for token_pos in curr_pos_ids[start_index:end_index+1]:
                        filter_pos = torch.where(1 == token_pos)
                        for f_p in filter_pos[0]:
                            if 2 < add_pos_cnt:
                                break
                            add_pos_ids[add_pos_cnt] = f_p.item()
                            add_pos_cnt += 1
