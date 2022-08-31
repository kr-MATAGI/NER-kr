### NE Tag Set
ETRI_TAG = {
    "O": 0,
    "B-PS": 1, "I-PS": 2,
    "B-LC": 3, "I-LC": 4,
    "B-OG": 5, "I-OG": 6,
    "B-AF": 7, "I-AF": 8,
    "B-DT": 9, "I-DT": 10,
    "B-TI": 11, "I-TI": 12,
    "B-CV": 13, "I-CV": 14,
    "B-AM": 15, "I-AM": 16,
    "B-PT": 17, "I-PT": 18,
    "B-QT": 19, "I-QT": 20,
    "B-FD": 21, "I-FD": 22,
    "B-TR": 23, "I-TR": 24,
    "B-EV": 25, "I-EV": 26,
    "B-MT": 27, "I-MT": 28,
    "B-TM": 29, "I-TM": 30
}

### Model List
NER_MODEL_LIST = {
    0: "NONE",
    1: "BERT",
    2: "ELECTRA",
    3: "RoBERTa",
    4: "BERT+LSTM(POS)+CRF",
    5: "RoBERTa+LSTM(POS)+CRF",
    6: "ELECTRA+LSTM(POS)+CRF",
    7: "ELECTRA_EOJEOL_MODEL",
    8: "ELECTRA_ALL_FEATURE_MODEL",
    9: "ELECTRA_CNNBiF_MODEL"
}