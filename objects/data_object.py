from typing import Callable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataObject:

    def __init__(self, x: [callable, pd.DataFrame], labels: pd.Series, converted: pd.Series):
        """
        Create data object
        Parameters
        ----------
        x: pd.DataFrame or callable
            Predictor values. If callable, the value is computed as x() once it is needed.
        labels: pd.Series
            Target values
        converted: pd.Series
            Current state: converted (True) or may still convert (False)
        """
        # All data must be equal in length
        if not callable(x):
            assert x.shape[0] == len(labels)
        assert len(labels) == len(converted)

        # Indices must be sorted
        assert (labels.index.sort_values() == labels.index).all()

        # All indices must be equal
        assert (labels.index == converted.index).all()

        # All converted datapoints must also have a positive label
        assert labels[converted].all()

        # All negative datapoints may not be converted
        assert not converted[~labels].any()

        self._x = x
        self.labels = labels
        self.converted = converted
        self.fix_index()

    def fix_index(self):
        if not callable(self._x):
            assert set(self._x.index) == set(self.labels.index)
            self._x = self._x.reindex(self.labels.index)

    @property
    def x(self) -> pd.DataFrame:
        if callable(self._x):
            self._x = self._x()
            self.fix_index()
        return self._x

    def train_test_split(self):
        """
        Create a 70-30 train-test stratified split

        Returns
        -------
        train_object: DataObject
            DataObject containing a stratified 70% training split of this DataObject
        test_object: DataObject
            DataObject containing a stratified 30% test split of this DataObject
        """
        # Note: sorting is needed, because train_test_split does not keep order
        # TODO: maybe make this lazy too, selecting a mask/indices only?
        arr = train_test_split(self.x, self.labels, self.converted,
                               train_size=0.7, random_state=0,
                               stratify=self.labels)
        arr = list(map(lambda df: df.reindex(df.index.sort_values()), arr))
        return type(self)(*arr[::2]), type(self)(*arr[1::2])

    def __add__(self, other):
        """
        Combine the data of DataObjects

        Parameters
        ----------
        other: DataObject
            The information to be added

        Returns
        -------
        new_data_object: DataObject
            The concatenated data
        """
        # Note: sorting is needed, because there is no guarantee about the other data
        assert isinstance(other, type(self))

        new_labels = pd.concat([self.labels, other.labels]).sort_index()
        new_converted = pd.concat([self.converted, other.converted]).sort_index()

        return type(self)(lambda: pd.concat([self.x, other.x]),
                          new_labels,
                          new_converted)

    def _split(self, mask):
        """
        Split this DataObject based on a boolean mask

        Parameters
        ----------
        mask: Iterable of bool
            boolean mask to indicate whether to keep each data point

        Returns
        -------
        masked_data_object: DataObject
            The subset of this DataObject that adheres to the mask
        """
        # Note: no sorting is needed, because the boolean mask keeps the sorting order
        return type(self)(
            lambda: self.x[mask],
            self.labels[mask],
            self.converted[mask],
        )

    @property
    def index(self):
        """
        The index of this DataObject

        Returns
        -------
        idx: pd.Index
            The index
        """
        return self.labels.index

    @property
    def converted_only(self):
        """
        Get a split of this DataObject with only the converted datapoints

        Returns
        -------
        converted_data_object: DataObject
            The new DataObject with only converted datapoints
        """
        return self._split(self.converted)

    @property
    def non_converted_only(self):
        """
        Get a split of this DataObject with only the non-converted datapoints

        Returns
        -------
        non_converted_data_object: DataObject
            The new DataObject with only non-converted datapoints
        """
        return self._split(~self.converted)

    @property
    def normalizer_class(self):
        """
        The class that is used to normalize this DataObject

        Returns
        -------
        obj: Any
            A Scaler object
        """
        return StandardScaler

    def normalize(self, normalizer: [Callable, None] = None) -> Tuple:
        """
        Get a new DataObject for which the x-values are scaled

        Parameters
        ----------
        normalizer: callable, None
            If None, use this DataObjects Scaler to scale the predictor data. If callable, use the callable to scale the
            predictor data.

        Returns
        -------
        scaled_data_object: DataObject
            The scaled DataObject. labels and converted are not changed
        normalizer: callable
            The normalizer used to normalizer. Input normalizer is returned if not None, otherwise a callable to fit
            to this DataObject
        """
        if normalizer is None:
            ss = self.normalizer_class()
            ss.fit(self.x)
            normalizer = ss.transform
        new_x = type(self.x)(pd.DataFrame(data=normalizer(self.x), columns=self.x.columns, index=self.x.index))
        return self._construct_new_from_x(new_x), normalizer

    def match_features(self, other):
        """
        Match features of another DataObject. Transforms predictor data to adhere to the other predictor columns.
        Missing features are set to 0, superfluous features are removed.

        Parameters
        ----------
        other: DataObject
            The other DataObject whose specification of predictor values to adhere to

        Returns
        -------
        matching_data_object: DataObject
            New DataObject matching the specification of other
        """
        assert isinstance(other, type(self))
        new_x = self.x.reindex(other.x.columns, axis='columns', fill_value=0)
        return self._construct_new_from_x(new_x)

    def _construct_new_from_x(self, new_x):
        return type(self)(new_x, self.labels, self.converted)
