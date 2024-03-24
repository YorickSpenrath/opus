from typing import Tuple

from .data_reference import DataReference


class SingleOpus:

    def __init__(self, ts: int, data_reference: DataReference, model_name: str, strategy: str):
        self.ts = ts
        self.data_reference = data_reference
        self.model_name = model_name
        self.strategy = strategy
        self.index = (ts, str(data_reference), model_name, strategy)

    def index(self) -> Tuple[int, str, str, str]:
        return self.ts, str(self.data_reference), self.model_name, self.strategy

    def __str__(self):
        return '|'.join(map(str, self.index))
