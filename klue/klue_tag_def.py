import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import json

KLUE_NER_TAG = {
    "B-PS": 0, "I-PS": 1, "B-LC": 2, "I-LC": 3, "B-OG": 4, "I-OG": 5,
    "B-DT": 6, "I-DT": 7, "B-TI": 8, "I-TI": 9, "B-QT": 10, "I-QT": 11,
    "O": 12,
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

@dataclass(frozen=True)
class NerFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self) -> None:
        return json.dumps(dataclasses.asdict(self)) + "\n"