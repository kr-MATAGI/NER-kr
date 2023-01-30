import random
import copy
import re
import numpy as np
import torch
import pickle

from pathlib import Path
from klue_tag_def import KLUE_NER_TAG, NerExample, NerFeatures
from utils.tag_def import MECAB_POS_TAG
from typing import List, Tuple, Union

from transformers import ElectraTokenizer
from eunjeon import Mecab
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

#===========================================================
class KlueSpanMaker:
#===========================================================
    def __init__(self, tokenizer_name, max_span_len: int = 8, max_seq_len: int = 128):
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len
        self.span_minus = int((max_span_len + 1) * max_span_len / 2)
        self.max_num_span = int(max_seq_len * max_span_len - self.span_minus)

    #===========================================================
    def create_span_npy_datasets(self, src_path: str, mode: str):
    #===========================================================
        print(f"[create_span_npy_datasets] src_path: {src_path}")

        examples, ori_examples, word_ne_pair = self.create_span_examples(src_path, mode)
        features = self.create_span_features(examples, mode=mode, label_list=KLUE_NER_TAG.keys(),
                                             word_ne_pair=word_ne_pair)

        all_input_ids = np.array([f.input_ids for f in features])
        all_attention_mask = np.array([f.attention_mask for f in features])
        all_token_type_ids = np.array(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features]
        )
        all_labels = np.array([f.label for f in features])

        assert self.max_seq_len == all_input_ids.shape[1], f"all_input_ids.len: {all_input_ids.shape[1]}"
        assert self.max_seq_len == all_attention_mask.shape[1], f"all_attn_mask.len: {all_attention_mask.shape[1]}"
        assert self.max_seq_len == all_token_type_ids.shape[1], f"all_token_type_ids.len: {all_token_type_ids.shape[1]}"
        assert self.max_seq_len == all_labels.shape[1], f"all_labels.len: {all_labels.shape[1]}"

        print(f"[create_span_npy_datasets] all_input_ids.shape: {all_input_ids.shape}")
        print(f"[create_span_npy_datasets] all_attention_mask.shape: {all_attention_mask.shape}")
        print(f"[create_span_npy_datasets] all_token_type_ids.shape: {all_token_type_ids.shape}")
        print(f"[create_span_npy_datasets] all_labels.shape: {all_labels.shape}")

        # Span NER
        span_only_label_token = np.array([f.span_only_label_token for f in features])
        all_span_len = np.array([f.all_span_len for f in features])
        real_span_mask_token = np.array([f.real_span_mask_token for f in features])
        all_span_idx = np.array([f.all_span_idx for f in features])

        assert self.max_num_span == span_only_label_token.shape[1], f"span_only_label_token.len: {span_only_label_token.shape[1]}"
        assert self.max_num_span == all_span_len .shape[1], f"all_span_len.len: {all_span_len.shape[1]}"
        assert self.max_num_span == real_span_mask_token.shape[1], f"real_span_mask_token.len: {real_span_mask_token.shape[1]}"
        assert self.max_num_span == all_span_idx.shape[1], f"all_span_idx.len: {all_span_idx.shape[1]}"

        print(f"[create_span_npy_datasets] span_only_label_token.shape: {span_only_label_token.shape}")
        print(f"[create_span_npy_datasets] all_span_len.shape: {all_span_len.shape}")
        print(f"[create_span_npy_datasets] real_span_mask_token.shape: {real_span_mask_token.shape}")
        print(f"[create_span_npy_datasets] all_span_idx.shape: {all_span_idx.shape}")

        np.save("../corpus/npy/klue_span_ner/" + mode + "_input_ids", all_input_ids)
        np.save("../corpus/npy/klue_span_ner/" + mode + "_attention_mask", all_attention_mask)
        np.save("../corpus/npy/klue_span_ner/" + mode + "_token_type_ids", all_token_type_ids)
        np.save("../corpus/npy/klue_span_ner/" + mode + "_label_ids", all_labels)

        np.save("../corpus/npy/klue_span_ner/" + mode + "_span_only_label_token", span_only_label_token)
        np.save("../corpus/npy/klue_span_ner/" + mode + "_all_span_len", all_span_len)
        np.save("../corpus/npy/klue_span_ner/" + mode + "_real_span_mask_token", real_span_mask_token)
        np.save("../corpus/npy/klue_span_ner/" + mode + "_all_span_idx", all_span_idx)

        print(f"[create_wordpiece_npy_datasets] Save ids - Complete !")

        # Save pickle
        with open("../corpus/npy/klue_span_ner/" + mode + "_origin.pkl", mode="wb") as ori_file:
            pickle.dump(ori_examples, ori_file)
            print(f"[create_wordpiece_npy_datasets] pickle len: {len(ori_examples)}")

    #===========================================================
    def create_span_features(self, examples, mode: str, label_list, word_ne_pair, task_mode: str="tagging"):
    #===========================================================
        if self.max_seq_len is None:
            max_length = self.tokenizer.max_len
        else:
            max_length = self.max_seq_len

        klue_tags = {
            "O": 0, "PS": 1, "LC": 2, "OG": 3,
            "DT": 4, "TI": 5, "QT": 6
        }
        ne_tag2ids = {k: v for k, v in klue_tags.items()}
        label_map = {label: i for i, label in enumerate(label_list)}
        print(f"[convert_wordpiece_features] label_map: \n{label_map}")

        def label_from_example(example: NerExample) -> Union[int, float, None, List[int]]:
            if example.label is None:
                return None
            if task_mode == "classification":
                return label_map[example.label]
            elif task_mode == "regression":
                return float(example.label)
            elif task_mode == "tagging":  # See KLUE paper: https://arxiv.org/pdf/2105.09680.pdf
                token_label = [label_map["O"]] * (max_length)
                for i, label in enumerate(example.label[: max_length - 2]):  # last [SEP] label -> 'O'
                    token_label[i + 1] = label_map[label]  # first [CLS] label -> 'O'
                return token_label
            raise KeyError(task_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = self.tokenizer(
            [(example.text_a, example.text_b) for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            if 0 == (i % 500):
                print(f"[create_span_features] {i} is Processing...")
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            # Span
            decode_wp = self.tokenizer.convert_ids_to_tokens(batch_encoding["input_ids"][i])
            all_span_idx_list = enumerate_spans(decode_wp, offset=0, max_span_width=max_span_len)
            all_span_len_list = []
            for idx_pair in all_span_idx_list:
                s_idx, e_idx = idx_pair
                span_len = e_idx - s_idx + 1
                all_span_len_list.append(span_len)

            span_idx_label_dict = self.make_span_idx_label_pair(word_ne_pair[i], decode_wp)
            span_idx_new_label_dict = self.convert2tokenIdx(span_idxLab=span_idx_label_dict,
                                                       all_span_idxs=all_span_idx_list)
            span_only_label_token = []  # 만들어진 span 집합들의 label
            for idx_str, label in span_idx_new_label_dict.items():
                span_only_label_token.append(ne_tag2ids[label])
            real_span_mask_token = np.ones_like(span_only_label_token).tolist()

            inputs.update({"all_span_idx": all_span_idx_list,
                           "all_span_len": all_span_len_list,
                           "span_only_label_token": span_only_label_token,
                           "real_span_mask_token": real_span_mask_token})

            feature = NerFeatures(**inputs, label=labels[i])
            features.append(feature)

        for i, example in enumerate(examples[:5]):
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("features: %s" % features[i])

        return features

    #===========================================================
    def create_span_examples(self, src_path: str, mode: str):
    #===========================================================
        print(f"[create_ner_examples] src_path: {src_path}")

        strip_char = "##"

        examples = []
        ori_examples = []
        file_path = Path(src_path)
        raw_text = file_path.read_text(encoding="utf-8").strip()
        raw_docs = re.split(r"\n\t?\n", raw_text)
        cnt = 0

        # for span
        all_sent_ne_pairs = []

        for doc in raw_docs:
            original_clean_tokens = []  # clean tokens (bert clean func)
            original_clean_labels = []  # clean labels (bert clean func)
            sentence = ""
            for line in doc.split("\n"):
                if line[:2] == "##":
                    guid = line.split("\t")[0].replace("##", "")
                    continue
                token, tag = line.split("\t")
                sentence += token
                if token == " ":
                    continue
                original_clean_tokens.append(token)
                original_clean_labels.append(tag)
            # sentence: "안녕 하세요.."
            # original_clean_labels: [안, 녕, 하, 세, 요, ., .]
            sent_words = sentence.split(" ")
            # sent_words: [안녕, 하세요..]
            modi_labels = []
            char_idx = 0
            for word in sent_words:
                # 안녕, 하세요
                correct_syllable_num = len(word)
                tokenized_word = self.tokenizer.tokenize(word)
                # case1: 음절 tokenizer --> [안, ##녕]
                # case2: wp tokenizer --> [안녕]
                # case3: 음절, wp tokenizer에서 unk --> [unk]
                # unk규칙 --> 어절이 통채로 unk로 변환, 단, 기호는 분리
                contain_unk = True if self.tokenizer.unk_token in tokenized_word else False
                for i, token in enumerate(tokenized_word):
                    token = token.replace(strip_char, "")
                    if not token:
                        modi_labels.append("O")
                        continue
                    modi_labels.append(original_clean_labels[char_idx])
                    if not contain_unk:
                        char_idx += len(token)
                if contain_unk:
                    char_idx += correct_syllable_num

            text_a = sentence  # original sentence
            examples.append(NerExample(guid=guid, text_a=text_a, label=modi_labels))
            ori_examples.append({"original_sentence": text_a, "original_clean_labels": original_clean_labels})
            cnt += 1

            ne_word_label_pair = []  # [ word, label ]
            curr_word = ""
            curr_label = ""
            for char_tok, char_label in zip(original_clean_tokens, original_clean_labels):
                if "B-" in char_label:
                    if 0 < len(curr_word) and 0 < len(curr_label):
                        ne_word_label_pair.append([curr_word, curr_label])
                    curr_word = char_tok
                    curr_label = char_label.replace("B-", "")
                elif "O" in char_label and "I-" not in char_label:
                    if 0 < len(curr_word) and 0 < len(curr_label):
                        ne_word_label_pair.append([curr_word, curr_label])
                    curr_word = ""
                    curr_label = ""
                else:
                    curr_word += char_tok
            if 0 < len(curr_word) and 0 < len(curr_label):
                ne_word_label_pair.append([curr_word, curr_label])

            all_sent_ne_pairs.append(ne_word_label_pair)

        return examples, ori_examples, all_sent_ne_pairs

    #=======================================================================================
    def make_span_idx_label_pair(self, ne_list, text_tokens):
    #=======================================================================================
        ret_dict = {}

        # print(text_tokens)
        b_check_use = [False for _ in range(len(text_tokens))]
        for ne_idx, ne_item in enumerate(ne_list):
            ne_char_list = list(ne_item[0].replace(" ", ""))

            concat_item_list = []
            for tok_idx in range(len(text_tokens)):
                if b_check_use[tok_idx]:
                    continue
                for sub_idx in range(tok_idx + 1, len(text_tokens)):
                    concat_word = ["".join(x).replace("##", "") for x in text_tokens[tok_idx:sub_idx]]
                    concat_item_list.append(("".join(concat_word), (tok_idx, sub_idx - 1)))  # Modify -1
            concat_item_list = [x for x in concat_item_list if "".join(ne_char_list) in x[0]]
            concat_item_list.sort(key=lambda x: len(x[0]))
            # print(ne_item.text, ne_item.type, concat_item_list)
            if 0 >= len(concat_item_list):
                continue
            target_idx_pair = concat_item_list[0][1]
            for bio_idx in range(target_idx_pair[0], target_idx_pair[1] + 1):
                b_check_use[bio_idx] = True
            key = str(target_idx_pair[0]) + ";" + str(target_idx_pair[1])
            ret_dict[key] = ne_item[1]

        # print(ret_dict)
        return ret_dict

    #=======================================================================================
    def convert2tokenIdx(self, all_span_idxs, span_idxLab):
    #=======================================================================================
        # convert the all the span_idxs from word-level to token-level
        sidxs = [x1 for (x1, x2) in all_span_idxs]
        eidxs = [x2 for (x1, x2) in all_span_idxs]

        span_idxs_new_label = {}
        for ns, ne, ose in zip(sidxs, eidxs, all_span_idxs):
            os, oe = ose
            oes_str = "{};{}".format(os, oe)
            nes_str = "{};{}".format(ns, ne)
            if oes_str in span_idxLab:
                label = span_idxLab[oes_str]
                span_idxs_new_label[nes_str] = label
            else:
                span_idxs_new_label[nes_str] = 'O'

        return span_idxs_new_label

#===========================================================
class KlueWordpieceMaker:
#===========================================================
    def __init__(self, tokenizer_name):
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    #===========================================================
    def create_wordpiece_npy_datasets(self, src_path: str, mode: str, max_length=510):
    #===========================================================
        print(f"[create_wordpiece_npy_datasets] src_path: {src_path}")

        examples, ori_examples = self.create_wordpiece_examples(src_path, mode)
        features = self.convert_wordpiece_features(examples, label_list=self.get_labels(), max_length=max_length)

        all_input_ids = np.array([f.input_ids for f in features])
        all_attention_mask = np.array([f.attention_mask for f in features])
        all_token_type_ids = np.array(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features]
        )
        all_labels = np.array([f.label for f in features])
        all_pos_flag = np.array([f.pos_flag for f in features])

        assert max_length == all_input_ids.shape[1], f"all_input_ids.len: {all_input_ids.shape[1]}"
        assert max_length == all_attention_mask.shape[1], f"all_attn_mask.len: {all_attention_mask.shape[1]}"
        assert max_length == all_token_type_ids.shape[1], f"all_token_type_ids.len: {all_token_type_ids.shape[1]}"
        assert max_length == all_labels.shape[1], f"all_labels.len: {all_labels.shape[1]}"
        assert max_length == all_pos_flag.shape[1], f"all_pos_flag.len: {all_pos_flag.shape[1]}"

        # Save Tensor
        print(f"[create_wordpiece_npy_datasets] all_input_ids.shape: {all_input_ids.shape}")
        print(f"[create_wordpiece_npy_datasets] all_attention_mask.shape: {all_attention_mask.shape}")
        print(f"[create_wordpiece_npy_datasets] all_token_type_ids.shape: {all_token_type_ids.shape}")
        print(f"[create_wordpiece_npy_datasets] all_labels.shape: {all_labels.shape}")
        print(f"[create_wordpiece_npy_datasets] all_pos_flag.shape: {all_pos_flag.shape}")

        np.save("../corpus/npy/klue_ner/" + mode + "_input_ids", all_input_ids)
        np.save("../corpus/npy/klue_ner/" + mode + "_attention_mask", all_attention_mask)
        np.save("../corpus/npy/klue_ner/" + mode + "_token_type_ids", all_token_type_ids)
        np.save("../corpus/npy/klue_ner/" + mode + "_label_ids", all_labels)
        np.save("../corpus/npy/klue_ner/" + mode + "_pos_ids", all_pos_flag)

        print(f"[create_wordpiece_npy_datasets] Save ids - Complete !")

        # Save pickle
        with open("../corpus/npy/klue_ner/" + mode + "_origin.pkl", mode="wb") as ori_file:
            pickle.dump(ori_examples, ori_file)
            print(f"[create_wordpiece_npy_datasets] pickle len: {len(ori_examples)}")

    #===========================================================
    def create_wordpiece_examples(self, src_path: str, mode: str):
    #===========================================================
        print(f"[create_wordpiece_examples] {src_path}")

        tokenizer = self.tokenizer
        strip_char = "##"

        examples = []
        ori_examples = []
        file_path = Path(src_path)
        raw_text = file_path.read_text(encoding="utf-8").strip()
        raw_docs = re.split(r"\n\t?\n", raw_text)
        cnt = 0

        for doc in raw_docs:
            original_clean_tokens = []  # clean tokens (bert clean func)
            original_clean_labels = []  # clean labels (bert clean func)
            sentence = ""
            for line in doc.split("\n"):
                if line[:2] == "##":
                    guid = line.split("\t")[0].replace("##", "")
                    continue
                token, tag = line.split("\t")
                sentence += token
                if token == " ":
                    continue
                original_clean_tokens.append(token)
                original_clean_labels.append(tag)
            # sentence: "안녕 하세요.."
            # original_clean_labels: [안, 녕, 하, 세, 요, ., .]
            sent_words = sentence.split(" ")
            # sent_words: [안녕, 하세요..]
            modi_labels = []
            char_idx = 0
            for word in sent_words:
                # 안녕, 하세요
                correct_syllable_num = len(word)
                tokenized_word = tokenizer.tokenize(word)
                # case1: 음절 tokenizer --> [안, ##녕]
                # case2: wp tokenizer --> [안녕]
                # case3: 음절, wp tokenizer에서 unk --> [unk]
                # unk규칙 --> 어절이 통채로 unk로 변환, 단, 기호는 분리
                contain_unk = True if tokenizer.unk_token in tokenized_word else False
                for i, token in enumerate(tokenized_word):
                    token = token.replace(strip_char, "")
                    if not token:
                        modi_labels.append("O")
                        continue
                    modi_labels.append(original_clean_labels[char_idx])
                    if not contain_unk:
                        char_idx += len(token)
                if contain_unk:
                    char_idx += correct_syllable_num

            text_a = sentence  # original sentence
            examples.append(NerExample(guid=guid, text_a=text_a, label=modi_labels))
            ori_examples.append({"original_sentence": text_a, "original_clean_labels": original_clean_labels})
            cnt += 1

        return examples, ori_examples

    #===========================================================
    def convert_wordpiece_features(self, examples: List[NerExample],
                                   label_list, max_length, task_mode="tagging") -> List[NerFeatures]:
    #===========================================================
        print(f"[convert_wordpiece_features] examples: {len(examples)}")

        if max_length is None:
            max_length = self.tokenizer.max_len

        label_map = {label: i for i, label in enumerate(label_list)}
        print(f"[convert_wordpiece_features] label_map: \n{label_map}")

        def label_from_example(example: NerExample) -> Union[int, float, None, List[int]]:
            if example.label is None:
                return None
            if task_mode == "classification":
                return label_map[example.label]
            elif task_mode == "regression":
                return float(example.label)
            elif task_mode == "tagging":  # See KLUE paper: https://arxiv.org/pdf/2105.09680.pdf
                token_label = [label_map["O"]] * (max_length)
                for i, label in enumerate(example.label[: max_length - 2]):  # last [SEP] label -> 'O'
                    token_label[i + 1] = label_map[label]  # first [CLS] label -> 'O'
                return token_label
            raise KeyError(task_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = self.tokenizer(
            [(example.text_a, example.text_b) for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            pos_flag = self.make_pos_flag(batch_encoding["input_ids"][i], examples[i].text_a)
            inputs.update({"pos_flag": pos_flag})

            feature = NerFeatures(**inputs, label=labels[i])
            features.append(feature)

        for i, example in enumerate(examples[:5]):
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("features: %s" % features[i])

        return features

    #===========================================================
    def get_labels(self) -> List[str]:
    #===========================================================
        return KLUE_NER_TAG.keys()

    #===========================================================
    def make_POS_results_by_mecab(self, example):
    #===========================================================
        mecab = Mecab()
        res_mecab = mecab.pos(example)

        # 어절별 글자 수 체크해서 띄워쓰기 적기
        split_text = example.split(" ")
        char_cnt_list = [len(st) for st in split_text]

        total_eojeol_morp = []
        use_check = [False for _ in range(len(res_mecab))]
        for char_cnt in char_cnt_list:
            curr_char_cnt = 0
            for ej_idx, eojeol in enumerate(res_mecab):
                if char_cnt == curr_char_cnt:
                    break
                if use_check[ej_idx]:
                    continue
                total_eojeol_morp.append([eojeol[0], eojeol[1].split("+")])
                curr_char_cnt += len(eojeol[0])
                use_check[ej_idx] = True

        return total_eojeol_morp

    #===========================================================
    def make_pos_flag(self, input_ids, example):
    #===========================================================
        mecab_tag2id = {v: k for k, v in MECAB_POS_TAG.items()}
        mecab_id2tag = {k: v for k, v in MECAB_POS_TAG.items()}

        mecab_res = self.make_POS_results_by_mecab(example)

        decode_wp = self.tokenizer.convert_ids_to_tokens(input_ids)
        pos_ids = [[mecab_tag2id["O"]] for _ in range(len(decode_wp))]
        b_check_pos_use = [False for _ in range(len(decode_wp))]
        for tok_pos in mecab_res:
            curr_pos = []
            for pos in tok_pos[1]:
                filter_pos = pos if "UNKNOWN" != pos and "NA" != pos and "UNA" != pos and "VSV" != pos else "O"
                curr_pos.append(mecab_tag2id[filter_pos])

            for root_idx in range(1, len(decode_wp)):
                if b_check_pos_use[root_idx]:
                    continue
                concat_item_list = []
                for tok_idx in range(root_idx, len(decode_wp)):
                    if b_check_pos_use[tok_idx]:
                        continue
                    for sub_idx in range(tok_idx, len(decode_wp)):
                        concat_word = ["".join(x).replace("##", "") for x in decode_wp[tok_idx:sub_idx + 1]]
                        concat_item_list.append(("".join(concat_word), (tok_idx, sub_idx)))
                concat_item_list = [x for x in concat_item_list if tok_pos[0] in x[0]]
                if 0 >= len(concat_item_list):
                    continue
                concat_item_list.sort(key=lambda x: len(x[0]))
                target_idx_pair = concat_item_list[0][1]
                b_check_pos_use[target_idx_pair[0]] = b_check_pos_use[target_idx_pair[1]] = True
                # print(concat_item_list[0][0], tok_pos[0], tok_pos[1])
                if concat_item_list[0][0] == tok_pos[0]:
                    pos_ids[target_idx_pair[0]] = curr_pos
                break
            # end loop, pos_ids

        # FOR TEST - pos ids debug
        # for p, d in zip(pos_ids, decode_wp):
        #     print(d, [mecab_id2tag[x] for x in p])
        # input()

        # POS One-hot
        pos_bit_flags = []
        for pos in pos_ids:
            curr_token_bit_flag = [0 for _ in range(len(MECAB_POS_TAG.keys()) - 1)]
            for p_id in pos:
                if 0 == p_id:
                    continue
                curr_token_bit_flag[p_id-1] = 1
            pos_bit_flags.append(curr_token_bit_flag)
        # end loop, pos_bit_flag

        # FOR TEST - pos ids debug
        # for p, d, b in zip(pos_ids, decode_wp, pos_bit_flags):
        #     print(d, [mecab_id2tag[x] for x in p], b)
        # input()

        return pos_bit_flags

#=======================================================================================
def convert_morp_connected_tokens(sent_lvl_pos: Tuple[str, str], src_text: str):
#=======================================================================================
    ret_conv_morp_tokens = []

    g_SYMBOL_TAGS = [
        "SF", "SE", # (마침표, 물음표, 느낌표), 줄임표
        "SSO", "SSC", # 여는 괄호, 닫는 괄호
        "SC", "SY", # 구분자, (붙임표, 기타 기호)
    ]

    # 어절별 글자 수 체크해서 띄워쓰기 적기
    split_text = src_text.split(" ")
    char_cnt_list = [len(st) for st in split_text]

    total_eojeol_morp = []
    use_check = [False for _ in range(len(sent_lvl_pos))]
    for char_cnt in char_cnt_list:
        eojeol_morp = []
        curr_char_cnt = 0
        for ej_idx, eojeol in enumerate(sent_lvl_pos):
            if char_cnt == curr_char_cnt:
                total_eojeol_morp.append(copy.deepcopy(eojeol_morp))
                eojeol_morp.clear()
                break
            if use_check[ej_idx]:
                continue
            eojeol_morp.append(eojeol)
            curr_char_cnt += len(eojeol[0])
            use_check[ej_idx] = True
        if 0 < len(eojeol_morp):
            total_eojeol_morp.append(eojeol_morp)

    for eojeol in total_eojeol_morp:
        for mp_idx, morp in enumerate(eojeol):  # morp (마케팅, NNG)
            if 0 == mp_idx or morp[1] in g_SYMBOL_TAGS:
                ret_conv_morp_tokens.append((morp[0], morp[1], False))
            else:
                if eojeol[mp_idx - 1][1] not in g_SYMBOL_TAGS:
                    conv_morp = (morp[0], morp[1], True)
                    ret_conv_morp_tokens.append(conv_morp)
                else:
                    ret_conv_morp_tokens.append((morp[0], morp[1], False))

    return ret_conv_morp_tokens

### MAIN ###
if "__main__" == __name__:
    print("[klue_parser] __MAIN__ !")

    train_data_path = "../corpus/klue/klue-ner-v1.1_train.tsv"
    dev_data_path = "../corpus/klue/klue-ner-v1.1_dev.tsv"

    making_mode = "wordpiece" # or span
    if "span" == making_mode:
        target_n_pos = 14
        target_tag_list = [  # NN은 NNG/NNP 통합
            "NNG", "NNP", "SN", "NNB", "NR", "NNBC",
            "JKS", "JKC", "JKG", "JKO", "JKB", "JX", "JC", "JKV", "JKQ",
        ]
        create_span_npy_datasets(src_path=dev_data_path, target_n_pos=target_n_pos, target_tag_list=target_tag_list, mode="dev")
        create_span_npy_datasets(src_path=train_data_path, target_n_pos=target_n_pos, target_tag_list=target_tag_list, mode="train")
    else:
        wp_maker = KlueWordpieceMaker(tokenizer_name="monologg/koelectra-base-v3-discriminator")
        wp_maker.create_wordpiece_npy_datasets(src_path=dev_data_path, mode="dev", max_length=128)
        wp_maker.create_wordpiece_npy_datasets(src_path=train_data_path, mode="train", max_length=128)