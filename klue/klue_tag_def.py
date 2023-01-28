import dataclasses
from dataclasses import dataclass
from typing import Optional
import json

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

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    def to_json_string(self) -> None:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"