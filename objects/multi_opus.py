from pathlib import Path

import matplotlib.pyplot as plt
import abc
from typing import List, Iterator

from opus.objects import AbstractOpus


class AbstractMultiOpus(abc.ABC):

    def __init__(self, _list: List[AbstractOpus]):
        self._list = _list

    @property
    def list_of_opus(self) -> List[AbstractOpus]:
        return self._list

    def get_cross(self, ao: AbstractOpus):
        if self.cross_mode == 'sequential':
            index = self.list_of_opus.index(ao)
            return self.list_of_opus[:index]
        elif self.cross_mode == 'others':
            return [opus for opus in self.list_of_opus if opus != ao]
        else:
            raise NotImplementedError(self.cross_mode)

    def __getitem__(self, item):
        return self.list_of_opus[item]

    @property
    @abc.abstractmethod
    def cross_mode(self) -> str:
        pass

    def __iter__(self) -> Iterator[AbstractOpus]:
        return iter(self.list_of_opus)

    @abc.abstractmethod
    def export_figure(self, f: plt.Figure, name: str):
        pass


class FileBasedMultiOpus(AbstractMultiOpus, abc.ABC):

    def export_figure(self, f: plt.Figure, name: str):
        fn = self.fd / 'figures' / f'{name}.svg'
        fn.parent.mkdir(exist_ok=True, parents=True)
        f.savefig(fn)

    @property
    def fd(self) -> Path:
        # noinspection PyUnresolvedReferences
        return self.list_of_opus[-1].fd.parent
