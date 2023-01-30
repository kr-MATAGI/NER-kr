import logging
import numpy as np
import torch
import random
import pickle
import os

from utils.tag_def import NIKL_POS_TAG, MECAB_POS_TAG

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

# config, model
from transformers import ElectraConfig, AutoConfig, AutoModelForTokenClassification, ElectraForTokenClassification

from model.morp_electra_model import ELECTRA_MECAB_MORP
from model.span_ner_model import ElectraSpanNER

#===============================================================
def print_parameters(args, logger):
#===============================================================
    logger.info(f"ckpt_dir: {args.ckpt_dir}")
    logger.info(f"output_dir: {args.output_dir}")

    logger.info(f"train_npy: {args.train_npy}")
    logger.info(f"dev_npy: {args.dev_npy}")
    logger.info(f"test_npy: {args.test_npy}")

    logger.info(f"evaluate_test_during_training: {args.evaluate_test_during_training}")
    logger.info(f"eval_all_checkpoints: {args.eval_all_checkpoints}")

    logger.info(f"save_optimizer: {args.save_optimizer}")
    logger.info(f"do_lower_case: {args.do_lower_case}")

    logger.info(f"do_train: {args.do_train}")
    logger.info(f"do_eval: {args.do_eval}")

    logger.info(f"max_seq_len: {args.max_seq_len}")
    logger.info(f"num_train_epochs: {args.num_train_epochs}")

    logger.info(f"weight_decay: {args.weight_decay}")
    logger.info(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    logger.info(f"adam_epsilon: {args.adam_epsilon}")
    logger.info(f"warmup_proportion: {args.warmup_proportion}")

    logger.info(f"max_steps: {args.max_steps}")
    logger.info(f"max_grad_norm: {args.max_grad_norm}")
    logger.info(f"seed: {args.seed}")

    logger.info(f"model_name_or_path: {args.model_name_or_path}")
    logger.info(f"train_batch_size: {args.train_batch_size}")
    logger.info(f"eval_batch_size: {args.eval_batch_size}")
    logger.info(f"learning_rate: {args.learning_rate}")

    logger.info(f"logging_steps: {args.logging_steps}")
    logger.info(f"save_steps: {args.save_steps}")

#===============================================================
def load_corpus_span_ner_npy(src_path: str, mode: str="train"):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode


    input_ids = np.load(root_path + "_input_ids.npy")
    attention_mask = np.load(root_path + "_attention_mask.npy")
    token_type_ids = np.load(root_path + "_token_type_ids.npy")
    label_ids = np.load(root_path + "_label_ids.npy")

    all_span_idx = np.load(root_path + "_all_span_idx.npy")
    all_span_len = np.load(root_path + "_all_span_len_list.npy")
    real_span_mask = np.load(root_path + "_real_span_mask_token.npy")
    span_only_label = np.load(root_path + "_span_only_label_token.npy")

    print(f"{mode}.shape - "
          f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, token_type_ids: {token_type_ids.shape}, "
          f"label_ids: {label_ids.shape}"
          f"all_span_idx: {all_span_idx.shape}, all_span_len: {all_span_len.shape}, "
          f"real_span_mask: {real_span_mask.shape}, span_only_label: {span_only_label.shape}")

    ori_examples = None
    with open(root_path+"_origin.pkl", mode="rb") as ori_f:
        ori_examples = pickle.load(ori_f)

    ret_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "label_ids": label_ids,
        "all_span_idx": all_span_idx,
        "all_span_len": all_span_len,
        "real_span_mask": real_span_mask,
        "span_only_label": span_only_label
    }

    return ret_dict, ori_examples


#===============================================================
def load_corpus_npy_datasets(src_path: str, mode: str="train", dataset_type: str="klue"):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode

    input_ids = np.load(root_path + "_input_ids.npy")
    attention_mask = np.load(root_path + "_attention_mask.npy")
    token_type_ids = np.load(root_path + "_token_type_ids.npy")
    label_ids = np.load(root_path + "_label_ids.npy")

    ori_examples = None
    with open(root_path+"_origin.pkl", mode="rb") as ori_f:
        ori_examples = pickle.load(ori_f)

    print(f"[load_corpus_npy_datasets][{dataset_type}][{mode}] input_ids.shape: {input_ids.shape}")
    print(f"[load_corpus_npy_datasets][{dataset_type}][{mode}] attention_mask.shape: {attention_mask.shape}")
    print(f"[load_corpus_npy_datasets][{dataset_type}][{mode}] token_type_ids.shape: {token_type_ids.shape}")
    print(f"[load_corpus_npy_datasets][{dataset_type}][{mode}] label_ids.shape: {label_ids.shape}")
    print(f"[load_corpus_npy_datasets][{dataset_type}][{mode}] origin_examples.len: {len(ori_examples)}")

    ret_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "label_ids": label_ids,
    }
    return ret_dict, ori_examples

#===============================================================
def init_logger():
# ===============================================================
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    return logger

#===============================================================
def set_seed(args):
#===============================================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if "cuda" == args.device:
        torch.cuda.manual_seed_all(args.seed)

#===============================================================
def f1_pre_rec(labels, preds, is_ner=True):
#===============================================================
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds),
            "recall": seqeval_metrics.recall_score(labels, preds),
            "f1": seqeval_metrics.f1_score(labels, preds),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }

#===============================================================
def show_ner_report(labels, preds):
#===============================================================
    return seqeval_metrics.classification_report(labels, preds, digits=3)

#===============================================================
def load_ner_config_and_model(user_select: int, args, tag_dict):
#===============================================================
    config = None
    model = None

    if 1 == user_select:
        # ELECTRA+LSTM(MECAB)+CRF
        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                               num_labels=len(tag_dict.keys()),
                                               id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                               label2id={label: i for i, label in enumerate(tag_dict.keys())})

        config.num_pos_labels = len(MECAB_POS_TAG.keys())  # Mecab All POS
        config.max_seq_len = 128
    elif 2 == user_select:
        # ELECTRA SPAN NER
        # etri_tag_dict = {'O': 0,
        #                  'FD': 1, 'EV': 2, 'DT': 3, 'TI': 4, 'MT': 5,
        #                  'AM': 6, 'LC': 7, 'CV': 8, 'PS': 9, 'TR': 10,
        #                  'TM': 11, 'AF': 12, 'PT': 13, 'OG': 14, 'QT': 15}

        klue_tags_dict = {
            "O": 0,
            "PS": 1, "LC": 2, "OG": 3,
            "DT": 4, "TI": 5, "QT": 6
        }

        span_tag_list = klue_tags_dict.keys()
        print("SPAN_TAG_DICT: ", klue_tags_dict)
        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                               num_labels=len(span_tag_list),
                                               id2label={idx: label for label, idx in klue_tags_dict.items()})

    # model
    if 1 == user_select:
        # ELECTRA+LSTM(MECAB)+[CRF] (MOR)
        model = ELECTRA_MECAB_MORP.from_pretrained(args.model_name_or_path, config=config)
    elif 2 == user_select:
        # ELECTRA SPAN NER
        model = ElectraSpanNER.from_pretrained(args.model_name_or_path, config=config)

    return config, model

#===============================================================
def load_model_checkpoints(user_select, checkpoint):
#===============================================================
    # model
    model = None

    if 1 == user_select:
        # ELECTRA+LSTM(MECAB)+[CRF]
        model = ELECTRA_MECAB_MORP.from_pretrained(checkpoint)
    elif 2 == user_select:
        # Span NER (ELECTRA)
        model = ElectraSpanNER.from_pretrained(checkpoint)

    return model

### TEST
if "__main__" == __name__:
    print("[ner_utils][__main__]")