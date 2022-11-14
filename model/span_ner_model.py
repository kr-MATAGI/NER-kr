import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraPreTrainedModel

from model.crf_layer import CRF
from model.classifier.span_classifier import SingleLinearClassifier, MultiNonLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F

#=======================================================
class ElectraSpanNER(ElectraPreTrainedModel):
#=======================================================
    def __init__(self, config):
        super(ElectraSpanNER, self).__init__(config)

        # Init
        self.hidden_size = config.hidden_size
        self.n_class = config.num_labels
        self.ids2label = config.id2label
        self.label2ids = config.label2id
        self.etri_tags = config.etri_tags

        self.span_combi_mode = "x,y"
        self.token_len_emb_dim = 50
        self.max_span_width = 8 # 원래는 4, NE가 4넘는게 많을 듯 보여 10 (max_spanLen)
        self.max_seq_len = 128

        self.span_len_emb_dim = 100
        ''' morp는 origin에서 {'isupper': 1, 'islower': 2, 'istitle': 3, 'isdigit': 4, 'other': 5}'''
        # self.morp_emb_dim = 100
        
        ''' 원본 Git에서는 Method 적용 개수에 따라 달라짐 '''
        self.input_dim = self.hidden_size * 2 + self.token_len_emb_dim + self.span_len_emb_dim
        self.model_dropout = 0.1

        # loss and softmax
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none') # 이거 none이여야 compute loss 동작
        self.softmax = torch.nn.Softmax(dim=-1) # for predict

        print("self.max_span_width: ", self.max_span_width)
        print("self.tokenLen_emb_dim: ", self.token_len_emb_dim)
        print("self.input_dim: ", self.input_dim) # 1586

        # Model
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)

        ''' 이거 2개 사용안함
        self.start_outputs = nn.Linear(self.hidden_size, 1)
        self.end_outputs = nn.Linear(self.hidden_size, 1)
        '''

        self.endpoint_span_extractor = EndpointSpanExtractor(input_dim=self.hidden_size,
                                                             combination=self.span_combi_mode,
                                                             num_width_embeddings=self.max_span_width,
                                                             span_width_embedding_dim=self.token_len_emb_dim,
                                                             bucket_widths=True)

        ''' 이거 2개 사용안함 
        self.linear = nn.Linear(10, 1)
        self.score_func = nn.Softmax(dim=-1)
        '''
        self.span_embedding = MultiNonLinearClassifier(self.input_dim,
                                                       self.n_class,
                                                       self.model_dropout)

        self.span_len_embedding = nn.Embedding(self.max_span_width + 1, self.span_len_emb_dim, padding_idx=0)
        # self.morp_embedding = nn.Embedding(len(args.morph2idx_list) + 1, self.morph_emb_dim, padding_idx=0)


    #==============================================
    def forward(self,
                all_span_lens, all_span_idxs_ltoken, real_span_mask_ltoken, input_ids,
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
        all_span_rep = torch.cat((all_span_rep, span_len_rep), dim=-1)
        all_span_rep = self.span_embedding(all_span_rep) # [batch, n_span, n_class] : [64, 502, 16]
        predict_prob = self.softmax(all_span_rep)

        # For Test
        preds = self.get_predict(predicts=predict_prob, all_span_idxs=all_span_idxs_ltoken, label_ids=label_ids)
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
        # print(loss.shape)
        loss = loss.view(batch_size, n_span)
        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())
        loss = torch.mean(loss)
        # predict = self.softmax(all_span_rep)

        return loss

    #==============================================
    def get_predict(self, predicts, all_span_idxs, label_ids=None):
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
                is_break = False
                curr_idxs = [i for i in range(span_pair[0][0], span_pair[0][1] + 1)]
                for c_idx in curr_idxs:
                    if check_use_idx[c_idx]:
                        is_break = True
                        break
                if is_break:
                    continue
                else:
                    # print(span_pair[0], span_pair[1], self.ids2label[span_pair[-1]])
                    batch_pred_span.append((span_pair[0], span_pair[-1]))
                    for s_idx in range(span_pair[0][0], span_pair[0][1] + 1):
                        check_use_idx[s_idx] = True

            decoded_pred = [self.ids2label[0] for _ in range(self.max_seq_len)]
            for span, label in batch_pred_span:
                if 0 == label:
                    continue
                s_idx = span[0]
                e_idx = span[1] + 1
                for dec_idx in range(s_idx, e_idx):
                    if 0 == label:
                        decoded_pred[dec_idx] = self.ids2label[0]
                    elif dec_idx == s_idx:
                        decoded_pred[dec_idx] = "B-" + self.ids2label[label]
                    else:
                        decoded_pred[dec_idx] = "I-" + self.ids2label[label]
            decoded_batches.append(decoded_pred)
            # print("\n===================")
            # print(decoded_pred)
            # print(label_ids[batch_idx])
            # print("\n===================")
        # end loop, batch
        return decoded_batches