import pickle
import numpy as np
from typing import List, Dict, Any
import torch
from klue.klue_tag_def import KLUE_NER_TAG

import sklearn
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.scheme import IOB2

from transformers import ElectraTokenizer

import logging
logger = logging.getLogger(__name__)

#=============================================
def validation_epoch_end(
        tokenizer_name,
        list_of_subword_preds: List[torch.Tensor],
        origin_datasets: List,
        max_seq_length: int = 510
):
#=============================================
    tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
    strip_char = "##"
    in_unk_token = "[+UNK]"

    original_examples = origin_datasets
    list_of_character_preds = []
    list_of_originals = []
    label_list = [k for k in KLUE_NER_TAG.keys()]

    for i, (subword_preds, example) in enumerate(zip(list_of_subword_preds, original_examples)):
        original_sentence = example["original_sentence"]  # 안녕 하세요 ^^
        character_preds = [subword_preds[0].tolist()]  # [CLS]
        character_preds_idx = 1
        for word in original_sentence.split(" "):  # ['안녕', '하세요', '^^']
            if character_preds_idx >= max_seq_length - 1:
                break
            subwords = tokenizer.tokenize(word)  # 안녕 -> [안, ##녕] / 하세요 -> [하, ##세요] / ^^ -> [UNK]
            if tokenizer.unk_token in subwords:  # 뻥튀기가 필요한 case!
                unk_aligned_subwords = tokenizer_out_aligner(
                    tokenizer,
                    word, subwords, strip_char,
                    in_unk_token
                )  # [UNK] -> [UNK, +UNK]
                unk_flag = False
                for subword in unk_aligned_subwords:
                    if character_preds_idx >= max_seq_length - 1:
                        break
                    subword_pred = subword_preds[character_preds_idx].tolist()
                    subword_pred_label = label_list[subword_pred]
                    if subword == tokenizer.unk_token:
                        unk_flag = True
                        character_preds.append(subword_pred)
                        continue
                    elif subword == in_unk_token:
                        if subword_pred_label == "O":
                            character_preds.append(subword_pred)
                        else:
                            _, entity_category = subword_pred_label.split("-")
                            character_pred_label = "I-" + entity_category
                            character_pred = label_list.index(character_pred_label)
                            character_preds.append(character_pred)
                        continue
                    else:
                        if unk_flag:
                            character_preds_idx += 1
                            subword_pred = subword_preds[character_preds_idx].tolist()
                            character_preds.append(subword_pred)
                            unk_flag = False
                        else:
                            character_preds.append(subword_pred)
                            character_preds_idx += 1  # `+UNK`가 끝나는 시점에서도 += 1 을 해줘야 다음 label로 넘어감
            else:
                for subword in subwords:
                    if character_preds_idx >= max_seq_length - 1:
                        break
                    subword = subword.replace(strip_char, "")  # xlm roberta: "▁" / others "##"
                    subword_pred = subword_preds[character_preds_idx].tolist()
                    subword_pred_label = label_list[subword_pred]
                    for i in range(0, len(subword)):  # 안, 녕
                        if i == 0:
                            character_preds.append(subword_pred)
                        else:
                            if subword_pred_label == "O":
                                character_preds.append(subword_pred)
                            else:
                                _, entity_category = subword_pred_label.split("-")
                                character_pred_label = "I-" + entity_category
                                character_pred = label_list.index(character_pred_label)
                                character_preds.append(character_pred)
                    character_preds_idx += 1
        character_preds.append(subword_preds[-1].tolist())  # [SEP] label
        list_of_character_preds.extend(character_preds)
        original_labels = ["O"] + example["original_clean_labels"][: len(character_preds) - 2] + ["O"]
        originals = []
        for label in original_labels:
            originals.append(label_list.index(label))
        assert len(character_preds) == len(originals)
        list_of_originals.extend(originals)

    entity_f1 = klue_ner_entity_macro_f1(list_of_character_preds, list_of_originals, label_list)
    char_f1 = klue_ner_char_macro_f1(list_of_character_preds, list_of_originals, label_list)

    return entity_f1, char_f1

#=============================================
def tokenizer_out_aligner(tokenizer, t_in: str, t_out: List[str], strip_char: str = "##", in_unk_token: str = "[+UNK]") -> List[str]:
#=============================================
    """Aligns with character-level labels after tokenization.

    Example:
        >>> t_in = "베쏭이,제5원소"
        >>> t_out = ['[UNK]', ',', '제', '##5', '##원', '##소']
        >>> tokenizer_out_aligner(t_in, t_out, strip_char="##")
        ['[UNK]', '[+UNK]', '[+UNK]', ',', '제', '##5', '##원', '##소']

        >>> t_in = "미나藤井美菜27가"
        >>> t_out = ['미나', '[UNK]', '[UNK]', '美', '[UNK]', '27', '##가']
        >>> tokenizer_out_aligner(t_in, t_out, strip_char="##")
        ['미나', '[UNK]', '[UNK]', '美', '[UNK]', '27', '##가']
    """
    t_out_new = []
    i, j = 0, 0
    UNK_flag = False
    while True:
        if i == len(t_in) and j == len(t_out) - 1:
            break
        step_t_out = len(t_out[j].replace(strip_char, "")) if t_out[j] != tokenizer.unk_token else 1
        if UNK_flag:
            t_out_new.append(in_unk_token)
        else:
            t_out_new.append(t_out[j])
        if j < len(t_out) - 1 and t_out[j] == tokenizer.unk_token and t_out[j + 1] != tokenizer.unk_token:
            i += step_t_out
            UNK_flag = True
            if t_in[i] == t_out[j + 1][0]:
                j += 1
                UNK_flag = False
        else:
            i += step_t_out
            j += 1
            UNK_flag = False
        if j == len(t_out):
            UNK_flag = True
            j -= 1
    return t_out_new


### Metrics
#=============================================
def klue_ner_char_macro_f1(preds: np.ndarray, labels: np.ndarray, label_list: List[str]) -> Any:
#=============================================
    """KLUE-NER character level macro f1 (except O tag)"""
    label_indices = list(range(len(label_list)))
    preds = np.array(preds).flatten().tolist()
    trues = np.array(labels).flatten().tolist()
    return sklearn.metrics.f1_score(trues, preds, labels=label_indices, average="macro", zero_division=True) * 100.0

#=============================================
def klue_ner_entity_macro_f1(preds: np.ndarray, labels: np.ndarray, label_list: List[str]) -> Any:
#=============================================t
    """KLUE-NER entity-level macro F1 (except O tag)"""
    preds = np.array(preds).flatten().tolist()
    labels = np.array(labels).flatten().tolist()
    preds_label = []
    labels_label = []

    for pred in preds:
        preds_label.append(label_list[pred])
    for label in labels:
        labels_label.append(label_list[label])

    entity_macro_f1 = ner_f1_score([labels_label], [preds_label], average="macro", mode="strict", scheme=IOB2)
    return entity_macro_f1 * 100.0


## MAIN ##
if "__main__" == __name__:
    validation_epoch_end(outputs=None, origin_datasets=None)