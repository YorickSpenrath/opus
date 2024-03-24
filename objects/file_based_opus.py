import abc
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..objects.abstract_opus import AbstractOpus


class FileBasedOpus(AbstractOpus, abc.ABC):

    def __init__(self, fd, store_models: bool = True):
        self.fd = fd
        self._store_models = store_models

    @staticmethod
    def _optional_create(fn: Path, create=False):
        if create:
            fn.parent.mkdir(parents=True, exist_ok=True)
        return fn

    @staticmethod
    def _remove_file(fn: Path):
        fn.unlink(missing_ok=True)

    def _fn_df(self, name, create=False) -> Path:
        return self._optional_create(self.fd / f'{name}.csv', create)

    def store_dataframe(self, df: pd.DataFrame, name: str):
        df.to_csv(self._fn_df(name, True))

    def load_dataframe(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self._fn_df(name))

    def remove_dataframe(self, name: str):
        self._remove_file(self._fn_object(name))

    def has_dataframe(self, name: str):
        return self._fn_df(name).exists()

    def _fn_object(self, name, create=False):
        return self._optional_create(self.fd / name, create)

    def store_object(self, obj: Any, name: str):
        with open(self._fn_object(name, True), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_object(self, name: str) -> Any:
        with open(self._fn_object(name), 'rb') as f:
            return pickle.load(f)

    def remove_object(self, name: str):
        self._remove_file(self._fn_object(name))

    def has_object(self, name: str) -> bool:
        return self._fn_object(name).exists()

    def _fn_picture(self, name: str) -> Path:
        return self.fd / f'{name}.svg'

    def store_picture(self, figure: plt.Figure, name: str):
        plt.savefig(figure, self._fn_picture(name))
        plt.close(figure)
