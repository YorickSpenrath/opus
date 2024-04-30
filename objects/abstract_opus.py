import abc
from itertools import chain, product
from typing import Iterator, Dict, Callable, Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC as SVC_CLASSIFIER
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import research.patterns.strings as ps
from .single_opus import SingleOpus
from .. import strings as ps
from ..objects.data_reference import DataReference
from ..ops import nape_from_numbers, nape_from_labels


class AbstractOpus:

    @abc.abstractmethod
    def has_dataframe(self, name: str):
        pass

    @abc.abstractmethod
    def store_dataframe(self, df: pd.DataFrame, name: str):
        pass

    @abc.abstractmethod
    def load_dataframe(self, name: str) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def remove_dataframe(self, name: str):
        pass

    def has_series(self, name: str) -> bool:
        return self.has_dataframe(name)

    def store_series(self, sr: pd.Series, name: str):
        self.store_dataframe(sr.to_frame(), name)

    def load_series(self, name: str) -> pd.Series:
        df = self.load_dataframe(name)
        return df.set_index(df.columns[0])[df.columns[1]]

    def remove_series(self, name: str):
        self.remove_series(name)

    @abc.abstractmethod
    def has_object(self, name: str) -> bool:
        pass

    @abc.abstractmethod
    def store_object(self, obj: Any, name: str):
        pass

    @abc.abstractmethod
    def load_object(self, name: str) -> Any:
        pass

    @abc.abstractmethod
    def remove_object(self, name: str):
        pass

    @abc.abstractmethod
    def store_picture(self, figure: plt.Figure, name: str):
        pass

    def __eq__(self, other):
        return str(self) == str(other)

    @abc.abstractmethod
    def __str__(self):
        """
        String representation of this Opus experiment

        Returns
        -------
        s: str
            String representation
        """
        pass

    ####################################################################################################################
    #                                           SINGLES GENERATION                                                     #
    ####################################################################################################################
    @abc.abstractmethod
    def data_generator(self, t: int, strategy: str) -> Iterator[DataReference]:
        """
        Generates DataReferences for the given timestamp

        Returns
        -------
        gen: Iterator[DataReference]
            The generator
        """
        pass

    @property
    def model_constructor_dict(self) -> Dict[str, Callable]:
        """
        Dictionary of all prediction models with construction factories

        Returns
        -------
        d: dict
            model_name, model_factory pairs
        """
        ret = {
            ps.KNN: lambda: KNeighborsClassifier(),
            ps.DECISION_TREE: lambda: DecisionTreeClassifier(random_state=0),
            ps.SVC: lambda: SVC_CLASSIFIER(random_state=0, probability=True),
        }

        # noinspection PyUnresolvedReferences
        from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier

        class PandasAhot(HoeffdingAdaptiveTreeClassifier):
            def fit(self, X, y, classes=None, sample_weight=None):
                if classes is None:
                    classes = np.unique(y)
                super().fit(X, y, classes, sample_weight)

        ret[ps.AHOT] = lambda: PandasAhot(random_state=0)

        return ret

    @staticmethod
    def model_needs_normalized_data(model_name: str):
        """
        Flag whether a given model_name needs data to be normalized (like kNN)

        Parameters
        ----------
        model_name: str
            The model name to evaluate

        Returns
        -------
        needs_normalization: bool
            True if the model is assumed to contain normalized data, False otherwise
        """
        if model_name == ps.KNN:
            return True
        return False

    @property
    def all_timesteps(self):
        result = set()
        for strategy in self.strategies:
            result = {*result, *self.timesteps_per_strategy(strategy)}
        return sorted(result)

    @abc.abstractmethod
    def timesteps_per_strategy(self, strategy: str) -> Iterator[int]:
        pass

    @property
    def add_converted(self) -> bool:
        return True

    @property
    def num_experiments(self):

        def n_features_per_timestep_and_strategy(t, strategy):
            # TODO: use YIELD/Combine
            dg = self.data_generator(t, strategy)
            if hasattr(dg, '__len__'):
                return dg.__len__()
            else:
                return len(list(dg))

        def n_features_per_strategy(strategy):
            def n_features_per_timestep(t):
                return n_features_per_timestep_and_strategy(t, strategy)

            return sum(map(n_features_per_timestep, self.timesteps_per_strategy(strategy)))

        return sum(map(n_features_per_strategy, self.strategies)) * len(self.model_constructor_dict)

    @property
    def strategies(self) -> List[str]:
        return [ps.STATIC, ps.UPDATE]

    @property
    def experiment_index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples([e.index for e in self.experiment_iterator()], names=ps.idx)

    def experiment_product(self) -> Iterator[Tuple[int, DataReference, str]]:
        outer_self = self

        class IT(Iterator):

            def __next__(self):
                raise StopIteration()

            def __iter__(self):
                # TODO: use YIELD
                def generate_strategy(strategy):
                    def generate_ts(ts):
                        return product([ts], outer_self.data_generator(ts, strategy),
                                       outer_self.model_constructor_dict.keys(), [strategy])

                    return chain(*(generate_ts(ts_i) for ts_i in outer_self.timesteps_per_strategy(strategy)))

                return chain(*(generate_strategy(strategy_i) for strategy_i in outer_self.strategies))

            def __len__(self):
                return outer_self.num_experiments

        return IT()

    def experiment_iterator(self) -> Iterator[SingleOpus]:
        outer_self = self

        class IT(Iterator):

            def __len__(self):
                return outer_self.num_experiments

            def __iter__(self):
                for (ts, data_reference, model_name, strategy) in outer_self.experiment_product():
                    yield SingleOpus(ts, data_reference, model_name, strategy)

            def __next__(self):
                raise StopIteration()

        return IT()

    @property
    def experiment_dataframe_index(self):
        idx_expected = pd.MultiIndex.from_tuples(self.experiment_product(),
                                                 names=[ps.TIMESTEP, ps.FEATURE_INDEX, ps.ML_ALGORITHM, ps.STRATEGY])
        idx_expected = pd.DataFrame(index=idx_expected).sort_index().reset_index()
        idx_expected[ps.FEATURE_INDEX] = idx_expected[ps.FEATURE_INDEX].astype(str)
        idx_expected = pd.MultiIndex.from_frame(idx_expected)
        return idx_expected

    ####################################################################################################################
    #                                                     SINGLES                                                      #
    ####################################################################################################################
    @staticmethod
    def _single_root_name(e: SingleOpus) -> str:
        return f'01_singles/{e.ts}_{e.data_reference}_{e.model_name}_{e.strategy}'

    # Model ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _single_model_name(e: SingleOpus, phase: int):
        return f'{AbstractOpus._single_root_name(e)}/{AbstractOpus.tt_string(phase)}_model.obj'

    def _model_should_be_stored(self, e):
        # KNN and AHOT should not be stored. KNN is too large, AHOT does not really work.
        if e.model_name in [ps.KNN, ps.AHOT]:
            return False
        elif e.model_name in [ps.DECISION_TREE, ps.SVC]:
            return True
        else:
            raise NotImplementedError(f'Unknown model storage for {e.model_name}')

    def has_model(self, e: SingleOpus, phase: int) -> bool:
        return self.has_object(self._single_model_name(e, phase))

    def store_model(self, model, e: SingleOpus, phase: int):
        if self._model_should_be_stored(e):
            self.store_object(model, self._single_model_name(e, phase))

    def load_model(self, e: SingleOpus, phase: int):
        if self._model_should_be_stored(e):
            return self.load_object(self._single_model_name(e, phase))
        else:
            raise NotImplementedError(f'Model of type {e.model_name} is not stored')

    def remove_model(self, e: SingleOpus, phase: int):
        self.remove_object(self._single_model_name(e, phase))

    def remove_all_models(self):
        for e in self.experiment_iterator():
            self.remove_model(e, True)
            self.remove_model(e, False)
            self.remove_model(e, 3)

    # Probabilities ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _single_probabilities_name(e: SingleOpus, phase: int):
        return f'{AbstractOpus._single_root_name(e)}/{AbstractOpus.tt_string(phase)}_probabilities'

    def has_probabilities(self, e: SingleOpus, phase: int):
        return self.has_series(self._single_probabilities_name(e, phase))

    def store_probabilities(self, pr: pd.Series, e: SingleOpus, phase: int):
        self.store_series(pr, self._single_probabilities_name(e, phase))

    def load_probabilities(self, e: SingleOpus, phase: int) -> pd.Series:
        sr = self.load_series(self._single_probabilities_name(e, phase))
        sr.index = sr.index.astype(str)
        return sr

    def remove_probabilities(self, e: SingleOpus, phase: int):
        self.remove_series(self._single_probabilities_name(e, phase))

    # Features ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _single_features_name(e: SingleOpus) -> str:
        return f'{AbstractOpus._single_root_name(e)}/features'

    def has_features(self, e: SingleOpus) -> bool:
        return self.has_series(self._single_features_name(e))

    def store_features(self, features: pd.Series, e: SingleOpus):
        self.store_series(features, self._single_features_name(e))

    def load_features(self, e: SingleOpus) -> pd.Series:
        return self.load_series(self._single_features_name(e))

    def remove_features(self, e: SingleOpus):
        self.remove_series(self._single_features_name(e))

    ####################################################################################################################
    #                                          THRESHOLD PREDICTION DATA                                               #
    ####################################################################################################################
    _tpm_root = '02_tpm_models'
    _tpm_data_name = f'{_tpm_root}/tpm_data'

    def has_tpm_df(self) -> bool:
        return self.has_dataframe(self._tpm_data_name)

    def store_tpm_df(self, xy: pd.DataFrame):
        self.store_dataframe(xy, self._tpm_data_name)

    def load_tpm_df(self):
        return self.load_dataframe(self._tpm_data_name).set_index(ps.idx)

    def remove_tpm_df(self):
        self.remove_dataframe(self._tpm_data_name)

    ####################################################################################################################
    #                                            ONE HOT ENCODERS                                                      #
    ####################################################################################################################
    _ohe_name = f'{_tpm_root}/OHE.model'

    def has_tpm_ohe(self):
        return self.has_object(self._ohe_name)

    def store_tpm_ohe(self, ohe: OneHotEncoder):
        self.store_object(ohe, self._ohe_name)

    def load_tpm_ohe(self) -> OneHotEncoder:
        return self.load_object(self._ohe_name)

    def remove_tpm_ohe(self):
        self.remove_object(self._ohe_name)

    ####################################################################################################################
    #                                        THRESHOLD PREDICTION MODELS                                               #
    ####################################################################################################################
    @property
    def tpm_model_dict(self) -> Dict[str, Callable]:
        return {ps.DECISION_TREE: lambda: DecisionTreeRegressor(random_state=0),
                ps.RANDOM_FOREST: lambda: RandomForestRegressor(random_state=0)}

    @property
    def tpm_shorthand_dict(self) -> Dict[str, str]:
        return {ps.DECISION_TREE: 'DT',
                ps.RANDOM_FOREST: 'RF'}

    @staticmethod
    def _tpm_name(model_name: str, metric: str) -> str:
        return f'{AbstractOpus._tpm_root}/{model_name}_{metric}.model'

    def has_tpm(self, model_name: str, metric: str) -> bool:
        return self.has_object(name=self._tpm_name(model_name, metric))

    def store_tpm(self, obj: Any, model_name: str, metric: str):
        self.store_object(obj, self._tpm_name(model_name, metric))

    def load_tpm(self, model_name: str, metric: str) -> Any:
        return self.load_object(self._tpm_name(model_name, metric))

    def remove_tpm(self, model_name: str, metric: str):
        self.remove_object(self._tpm_name(model_name, metric))

    ####################################################################################################################
    #                                               THRESHOLDS                                                         #
    ####################################################################################################################
    # Thresholds that can be computed from only this OPUS experiment
    @property
    def default_thresholds_dict(self) -> Dict[str, Callable]:

        def get_ratio(use_non_converted):
            if use_non_converted:
                prob = 'ancp'
            else:
                prob = 'acp'

            def threshold(df, metric):
                r = df[f'{prob}_{self.tt_string(False)}'] / df[f'{prob}_{self.tt_string(True)}']
                return r * df[f'oth_{metric}_{self.tt_string(True)}']

            return threshold

        def get_difference(use_non_converted):
            if use_non_converted:
                prob = 'ancp'
            else:
                prob = 'acp'

            def threshold(df, metric):
                r = df[f'{prob}_{self.tt_string(False)}'] - df[f'{prob}_{self.tt_string(True)}']
                return r * df[f'oth_{metric}_{self.tt_string(True)}']

            return threshold

        return {
            # Baselines
            'previous': lambda x, metric: x[f'oth_{metric}_{self.tt_string(True)}'],
            '.5': lambda x, c: 0.5,
            '0': lambda x, c: 0,
            '1': lambda x, c: 1,
            f'mean-{self.tt_string(True)}': lambda x, c: x[f'oth_{c}_{self.tt_string(True)}'].mean(),
            f'mean-{self.tt_string(False)}': lambda x, c: x[f'oth_{c}_{self.tt_string(False)}'].mean(),

            # Comparison
            'optimal': lambda x, metric: x[f'oth_{metric}_{self.tt_string(False)}'],

            # Heuristic
            'ancpx': get_ratio(True),
            'acpx': get_ratio(False),
            'ancp+': get_difference(True),
            'acp+': get_difference(False)
        }

    @property
    def threshold_translations(self) -> Dict[str, str]:
        return {
            'previous': '$oth^t_{i,i}$',
            ps.OPTIMAL: '$oth^t_{i,j}$',
            **{f'a{n}cpx': rf'$\varnothing a{lu}p$' for n, lu in zip(['', 'n'], 'lu')},
            **{f'a{n}cp+': rf'$\Delta a{lu}p$' for n, lu in zip(['', 'n'], 'lu')},
            '0': '$True$',
            '1': '$False$',
            '.5': 'Regular',
            f'mean-{self.tt_string(True)}': r'$\mu-train-M^t(e)$',
            f'mean-{self.tt_string(False)}': r'$\mu-test-M^t(e)$',
            f'{ps.COMPETITOR}:{ps.BASIC_PU}': '$PU$',
            f'{ps.ALTERNATIVE}:{ps.CONVERTED_RATIO}': r'$\varnothing$ Converted',
            f'{ps.ALTERNATIVE}:{ps.PROBABILITY_SUM}': r'$\Sigma_e \/ M^t_i(e)$',
            ps.RETRAIN: ps.RETRAIN.capitalize()
        }

    @staticmethod
    def _thresholds_name(metric: str):
        return f'03_thresholds/{metric}'

    def has_thresholds(self, metric: str):
        return self.has_dataframe(name=self._thresholds_name(metric))

    def store_thresholds(self, thresholds: pd.DataFrame, metric: str):
        self.store_dataframe(thresholds, self._thresholds_name(metric))

    def load_thresholds(self, metric: str) -> pd.DataFrame:
        return self.load_dataframe(self._thresholds_name(metric)).set_index(ps.idx)

    def remove_thresholds(self, metric: str):
        self.remove_dataframe(self._thresholds_name(metric))

    ####################################################################################################################
    #                                                 SCORES                                                           #
    ####################################################################################################################
    def compute_alternative_method_scores(self) -> Dict[str, pd.DataFrame]:

        # NAPE on probability sum ======================================================================================
        proba_nape = pd.DataFrame(index=self.experiment_dataframe_index, columns=['prev', 'this']).sort_index()

        for experiment in self.experiment_iterator():
            experiment: SingleOpus

            n_true_previous = experiment.data_reference.phase_1_data.labels.sum()
            n_pred_previous = self.load_probabilities(experiment, phase=True).sum()
            proba_nape.loc[experiment.index, 'prev'] = nape_from_numbers(n_true_previous, n_pred_previous)

            n_true_this = experiment.data_reference.phase_2_data.labels.sum()
            n_pred_this = self.load_probabilities(experiment, phase=False).sum()
            proba_nape.loc[experiment.index, 'this'] = nape_from_numbers(n_true_this, n_pred_this)

        proba_nape = proba_nape.astype(float)

        res = proba_nape.loc[proba_nape.groupby(level=0)['prev'].idxmax(), 'this'].rename(
            ps.PROBABILITY_SUM).to_frame().reset_index(level=list(range(1, len(ps.idx))), drop=True)

        # NAPE on conversion ratio =====================================================================================
        for t in self.all_timesteps:
            # The strategy does not matter for this alternative method
            dr = next(iter(self.data_generator(t, ps.STATIC)))

            converted_previous = dr.phase_1_data.converted.sum()
            converted_this = dr.phase_2_data.converted.sum()
            total_previous = dr.phase_1_data.non_converted_only.labels.sum()
            n_true = dr.phase_2_data.non_converted_only.labels.sum()
            n_pred = converted_this / converted_previous * total_previous
            res.loc[t, ps.CONVERTED_RATIO] = nape_from_numbers(n_true, n_pred)

        return {ps.NAPE_CONVERTED_OBJECT_COUNT: res}

    @property
    def alternative_method_scores_completed(self) -> bool:
        # Check if exists at all
        if not self.has_score_type(ps.NAPE_CONVERTED_OBJECT_COUNT, ps.ALTERNATIVE):
            return False

        # If so load
        df = self.load_score_type(ps.NAPE_CONVERTED_OBJECT_COUNT, ps.ALTERNATIVE)

        # Check all timesteps are there
        if not set(df.index) == set(self.all_timesteps):
            return False

        # Check all methods are there
        if not set(df.columns) == {ps.PROBABILITY_SUM, ps.CONVERTED_RATIO}:
            return False

        # If nothing fails, we are done
        return True

    @staticmethod
    def _score_name(metric: str, score_type: str) -> str:
        fd = '04_' + {
            ps.OPUS: 'A',
            ps.ALTERNATIVE: 'B',
            ps.COMPETITOR: 'C'
        }[score_type] + f'_{score_type}_scores'

        return f'{fd}/{metric}'

    def has_score_type(self, metric: str, score_type: str) -> bool:
        return self.has_dataframe(self._score_name(metric, score_type))

    def store_score_type(self, metric: str, score_type: str, scores: pd.DataFrame):
        self.store_dataframe(scores, self._score_name(metric, score_type))

    def load_score_type(self, metric: str, score_type: str) -> pd.DataFrame:
        df = self.load_dataframe(self._score_name(metric, score_type))
        if score_type == ps.ALTERNATIVE:
            return df.set_index(ps.TIMESTEP)
        elif score_type == ps.OPUS:
            return df.set_index(ps.idx)
        elif score_type == ps.COMPETITOR:
            return df.set_index(ps.idx)
        else:
            raise NotImplementedError(score_type)

    def remove_score_type(self, metric: str, score_type: str):
        self.remove_dataframe(self._score_name(metric, score_type))

    def load_all_thresholds(self, m: str, return_best_previous_hp_per_timestamp=False):
        _, best = self.load_all_scores(m, True)
        thresholds = self.load_thresholds(m).loc[best].reset_index(level=list(range(1, len(ps.idx))), drop=True)
        if return_best_previous_hp_per_timestamp:
            return thresholds, best
        else:
            return thresholds

    def load_all_scores(self, m: str, return_best_previous_hp_per_timestamp=False):
        # Scores
        scores = self.load_score_type(metric=m, score_type=ps.OPUS)

        best_previous_per_timestamp = self.load_tpm_df().groupby(level=0)[f'{m}_{self.tt_string(1)}'].idxmax()
        best_previous_per_timestamp = best_previous_per_timestamp[~pd.isna(best_previous_per_timestamp)]
        best_previous_per_timestamp = best_previous_per_timestamp[best_previous_per_timestamp.apply(
            lambda x: x[0] in scores.reset_index(level=list(range(1, len(ps.idx)))).index.unique())]
        df = scores.loc[best_previous_per_timestamp].reset_index(level=list(range(1, len(ps.idx))), drop=True)

        # Get the alternative scores
        if self.has_score_type(m, ps.ALTERNATIVE):
            alt = self.load_score_type(m, ps.ALTERNATIVE).reindex(self.all_timesteps)
            alt.columns = map(lambda x: f'alternative:{x}', alt.columns)
            df = pd.concat([df, alt], axis=1)

        # Get Phase 3 scores
        tpm_df = self.load_tpm_df()
        if f'{m}_{self.tt_string(3)}' in tpm_df.columns:
            phase3_scores = tpm_df.groupby(level=0)[f'{m}_{self.tt_string(3)}'].max().rename(ps.RETRAIN)
            df = pd.concat([df, phase3_scores], axis=1)

        # Get the competitor scores
        if self.has_score_type(m, ps.COMPETITOR):
            comp_full = self.load_score_type(m, ps.COMPETITOR)

            def competitor_name(x):
                a, b = x.rsplit('_', 1)
                assert b in [self.tt_string(True), self.tt_string(False)]
                return a

            def compute_comp(c):
                # This gets the best score in episode 2
                # return comp_full[f'{c}_{self.tt_string(False)}'].groupby(ps.TIMESTEP).max().rename(c)
                idx = comp_full.groupby(level=0)[f'{c}_{self.tt_string(True)}'].idxmax().dropna()
                return comp_full.loc[idx, f'{c}_{self.tt_string(False)}'].reset_index(level=list(range(1, len(ps.idx))),
                                                                                      drop=True).rename(c)

            comp = pd.concat(map(compute_comp, set(map(competitor_name, comp_full.columns))), axis=1)
            comp.columns = map(lambda x: f'{ps.COMPETITOR}:{x}', comp.columns)
            df = pd.concat([df, comp], axis=1)

        if return_best_previous_hp_per_timestamp:
            return df, best_previous_per_timestamp
        else:
            return df

    ####################################################################################################################
    #                                                 FIGURES                                                          #
    ####################################################################################################################
    @staticmethod
    def _figure_name(metric: str) -> str:
        return f'05_figures/{metric}'

    def store_figure(self, figure: plt.Figure, metric: str):
        self.store_picture(figure, self._figure_name(metric))

    ####################################################################################################################
    #                                                AUXILIARY                                                         #
    ####################################################################################################################
    @property
    @abc.abstractmethod
    def timestep_name(self):
        pass

    @staticmethod
    def tt_string(phase: [int, bool]):
        # I later decided to add 'phase 3' as well, this is the retraining after the second episode
        # initially there was training/testing as phase 1/2
        if isinstance(phase, bool):
            if phase:
                phase = 1
            else:
                phase = 2
        elif isinstance(phase, int):
            pass
        else:
            raise NotImplementedError()
        assert 1 <= phase <= 3
        return f"phase{phase}"

    @property
    def metric_methods_dict(self) -> Dict[str, Callable]:
        def f1(y_true, y_pred):
            # Note, sk-learned fixed this in 1.3
            if (y_pred.sum() == 0) and (y_true.sum() == 0):
                return np.nan

            return f1_score(y_true=y_true, y_pred=y_pred)

        return {ps.ACCURACY: balanced_accuracy_score,
                ps.F1: f1,
                ps.NAPE_CONVERTED_OBJECT_COUNT: nape_from_labels}

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def get_alternative_labels(self, ts: int, data_reference: DataReference, model_name: str) \
            -> [None, pd.Series]:
        return None

    @abc.abstractmethod
    def translate_tpm(self, x):
        pass

    @property
    @abc.abstractmethod
    def short_name(self):
        pass

    def store_characteristic_figure(self, f, name):
        self.store_picture(f, f'99_characteristics/{name}')
