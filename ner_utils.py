import logging
import numpy as np
import torch
import random
import pickle
import os

from utils.tag_def import NIKL_POS_TAG

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

# config, model
from transformers import ElectraConfig, AutoConfig, AutoModelForTokenClassification, ElectraForTokenClassification

from model.electra_lstm_crf import ELECTRA_POS_LSTM
from model.bert_lstm_crf import BERT_LSTM_CRF
from model.roberta_lstm_crf import RoBERTa_LSTM_CRF
from model.electra_eojeol_model import Electra_Eojeol_Model
from model.electra_feature_model import Electra_Feature_Model
from model.CNNBiF_model import ELECTRA_CNNBiF_Model

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
def load_corpus_npy_datasets(src_path: str, mode: str="train"):
#===============================================================
    root_path = "/".join(src_path.split("/")[:-1]) + "/" + mode

    dataset_npy = np.load(src_path)
    # token_seq_len = np.load(root_path + "_token_seq_len.npy")
    pos_tag_npy = np.load(root_path + "_pos_tag.npy")
    labels_npy = np.load(root_path + "_labels.npy")
    eojeol_ids = np.load(root_path + "_eojeol_ids.npy")
    # ls_ids = np.load(root_path + "_ls_ids.npy")
    #entity_ids = np.load(root_path + "_entity_ids.npy")

    return dataset_npy, pos_tag_npy, labels_npy, eojeol_ids, #ls_ids, entity_ids

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
    return seqeval_metrics.classification_report(labels, preds)

#===============================================================
def load_ner_config_and_model(user_select: int, args, tag_dict):
#===============================================================
    config = None
    model = None

    # config
    if 1 == user_select:
        # BERT
        config = AutoConfig.from_pretrained("klue/bert-base",
                                            num_labels=len(tag_dict.keys()),
                                            id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                            label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.max_seq_len = 128
    elif 2 == user_select:
        # ELECTRA
        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                               num_labels=len(tag_dict.keys()),
                                               id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                               label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.max_seq_len = 128
    elif 3 == user_select:
        # RoBERTa
        config = AutoConfig.from_pretrained("klue/roberta-base",
                                            num_labels=len(tag_dict.keys()),
                                            id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                            label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.max_seq_len = 128
    elif 4 == user_select:
        # BERT+LSTM(POS)+CRF
        config = AutoConfig.from_pretrained("klue/bert-base",
                                            num_labels=len(tag_dict.keys()),
                                            id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                            label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49  # NIKL
        config.max_seq_len = 128
    elif 5 == user_select:
        # RoBERTa+LSTM(POS)+CRF
        config = AutoConfig.from_pretrained("klue/roberta-base",
                                            num_labels=len(tag_dict.keys()),
                                            id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                            label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49  # NIKL
        config.max_seq_len = 128
    elif 6 == user_select:
        # ELECTRA+LSTM(POS)+CRF
        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                               num_labels=len(tag_dict.keys()),
                                               id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                               label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49  # 국립국어원 형태 분석 말뭉치
        config.max_seq_len = 128
    elif 7 == user_select:
        # ELECTRA + Eojeol Embedding -> Transformer
        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                               num_labels=len(tag_dict.keys()),
                                               id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                               label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.model_name = "monologg/koelectra-base-v3-discriminator"
        config.num_pos_labels = len(NIKL_POS_TAG.keys())  # NIKL
        config.max_seq_len = 128
    elif 8 == user_select:
        # ELECTRA + ALL FEATURES (POS, Eojeol, Entity) -> Transformer + CRF
        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                               num_labels=len(tag_dict.keys()),
                                               id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                               label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49  # NIKL
        config.max_seq_len = 128
        config.max_eojeol_len = 50
    elif 9 == user_select:
        # ELECTRA + CNNBiF
        config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",
                                               num_labels=len(tag_dict.keys()),
                                               id2label={str(i): label for i, label in enumerate(tag_dict.keys())},
                                               label2id={label: i for i, label in enumerate(tag_dict.keys())})
        config.num_pos_labels = 49 # NIKL
        config.max_seq_len = 128
        config.max_eojeol_len = 50

    # model
    if 1 == user_select:
        # BERT-base
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
    elif 2 == user_select:
        # ELECTRA-base
        model = ElectraForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
    elif 3 == user_select:
        # RoBERTa-base
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, config=config)
    elif 4 == user_select:
        # BERT+LSTM(POS)+CRF
        model = BERT_LSTM_CRF.from_pretrained(args.model_name_or_path, config=config)
    elif 5 == user_select:
        # RoBERTa+LSTM(POS)+CRF
        model = RoBERTa_LSTM_CRF.from_pretrained(args.model_name_or_path, config=config)
    elif 6 == user_select:
        # ELECTRA+LSTM(POS)+CRF
        model = ELECTRA_POS_LSTM.from_pretrained(args.model_name_or_path, config=config)
    elif 7 == user_select:
        # ELECTRA + Eojeol Embedding -> Transformer + CRF
        model = Electra_Eojeol_Model.from_pretrained(args.model_name_or_path, config=config)
    elif 8 == user_select:
        # ELECTRA + ALL FEATURES (POS, Eojeol, Entity) -> Transformer + CRF
        model = Electra_Feature_Model.from_pretrained(args.model_name_or_path, config=config)
    elif 9 == user_select:
        # ELECTRA + CNNBiF
        model = ELECTRA_CNNBiF_Model.from_pretrained(args.model_name_or_path, config=config)

    return config, model

#===============================================================
def load_model_checkpoints(user_select, checkpoint):
#===============================================================
    # model
    model = None
    if 1 == user_select:
        # BERT-base
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    elif 2 == user_select:
        # ELECTRA-base
        model = ElectraForTokenClassification.from_pretrained(checkpoint)
    elif 3 == user_select:
        # RoBERTa-base
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    elif 4 == user_select:
        # BERT+LSTM(POS)+CRF
        model = BERT_LSTM_CRF.from_pretrained(checkpoint)
    elif 5 == user_select:
        # RoBERTa+LSTM(POS)+CRF
        model = RoBERTa_LSTM_CRF.from_pretrained(checkpoint)
    elif 6 == user_select:
        # ELECTRA+LSTM(POS)+CRF
        model = ELECTRA_POS_LSTM.from_pretrained(checkpoint)
    elif 7 == user_select:
        # ELECTRA + Eojeol Embedding -> Transformer + CRF
        model = Electra_Eojeol_Model.from_pretrained(checkpoint)
    elif 8 == user_select:
        # ELECTRA + ALL FEATURES (POS, Eojeol, Entity) -> Transformer + CRF
        model = Electra_Feature_Model.from_pretrained(checkpoint)
    elif 9 == user_select:
        # ELECTRA + CNNBiF
        model = ELECTRA_CNNBiF_Model.from_pretrained(checkpoint)

    return model

### TEST
if "__main__" == __name__:
    print("[ner_utils][__main__]")