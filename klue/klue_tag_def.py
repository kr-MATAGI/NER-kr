from dataclasses import dataclass
from typing import Optional

KLUE_NER_TAG = {
    "O": 0,
    "B-PS": 1, "I-PS": 2, "B-LC": 3, "I-LC": 4, "B-OG": 5, "I-OG": 6,
    "B-DT": 7, "I-DT": 8, "B-TI": 9, "I-TI": 10, "B-QT": 11, "I-QT": 12,
}

@dataclass
class NerExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None