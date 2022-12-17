from dataclasses import dataclass
from typing import Optional

KLUE_NER_TAG = [
    "B-PS", "I-PS", "B-LC", "I-LC", "B-OG", "I-OG",
    "B-DT", "I-DT", "B-TI", "I-TI", "B-QT", "I-QT",
    "O"
]

@dataclass
class NerExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None