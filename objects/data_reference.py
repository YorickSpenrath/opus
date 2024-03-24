import abc

from ..objects.data_object import DataObject


class DataReference(abc.ABC):

    def __repr__(self):
        return self.__str__()

    @abc.abstractmethod
    def __str__(self):
        """
        String value of this DataReference

        Returns
        -------
        s: str
            String value
        """
        pass

    @property
    def encoding(self):
        """
        Encoding of this DataReference. Useful if you want to group similar DataReferences.

        Returns
        -------
        enc: str
            Encoding
        """
        return str(self)

    @property
    @abc.abstractmethod
    def phase_1_data(self) -> DataObject:
        """
        DataObject for Phase 1

        Returns
        -------
        do_p1: DataObject
            DataObject for Phase 1
        """
        pass

    @property
    @abc.abstractmethod
    def phase_2_data(self) -> DataObject:
        """
        DataObject for Phase 2

        Returns
        -------
        do_p2: DataObject
            DataObject for Phase 2
        """
        pass
