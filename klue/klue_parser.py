import re
from pathlib import Path
from klue_tag_def import KLUE_NER_TAG, NerExample

from transformers import ElectraTokenizer

#===========================================================
def create_ner_examples(src_path: str, tokenizer):
#===========================================================
    print(f"[create_ner_examples] src_path: {src_path}")

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

        for char_tok, char_label in zip(original_clean_tokens, original_clean_labels):
            

        return examples

#===========================================================
def create_features(examples, tokenizer, max_seq_len: int = 128):
#===========================================================
    print(f"[create_features] Label list: {KLUE_NER_TAG}")

    label_map = {label: i for i, label in enumerate(KLUE_NER_TAG)}
    labels = [label_from_example(example, label_map, max_seq_len) for example in examples]
    print(labels)

#===========================================================
def label_from_example(example: NerExample, label_map, max_length):
#===========================================================
    token_label = [label_map["O"]] * (max_length)
    for i, label in enumerate(example.label[: max_length - 2]):  # last [SEP] label -> 'O'
        token_label[i + 1] = label_map[label]  # first [CLS] label -> 'O'
    return token_label

#===========================================================
def create_npy_datasets(src_path: str):
#===========================================================
    print(f"[create_npy_datasets] src_path: {src_path}")

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    examples = create_ner_examples(src_path, tokenizer)
    features = create_features(examples, tokenizer, 128)

### MAIN ###
if "__main__" == __name__:
    print("[klue_parser] __MAIN__ !")

    train_data_path = "./data/klue-ner-v1.1_train.tsv"
    dev_data_path = "./data/klue-ner-v1.1_dev.tsv"
    create_npy_datasets(src_path=dev_data_path)