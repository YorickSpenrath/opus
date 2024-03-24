import numpy as np
import pandas as pd
from pulearn import ElkanotoPuClassifier

from ._decorators import run_all
from ..execution.compute1_A_probabilities import get_positive_probabilities
from ..objects import AbstractOpus
from .. import strings as ps
from ..objects.multi_opus import AbstractMultiOpus
from ..objects.single_opus import SingleOpus


class AdaptedElkanotoPuClassifier(ElkanotoPuClassifier):

    def __init__(self, estimator, hold_out_ratio=0.1, random_state=None):
        super().__init__(estimator, hold_out_ratio)
        self.rng = np.random.RandomState(seed=random_state)

    def fit(self, X, y):
        """Fits the classifier

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        positives = np.where(y == 1.0)[0]
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))
        # check for the required number of positive examples
        if len(positives) <= hold_out_size:
            raise ValueError(
                'Not enough positive examples to estimate p(s=1|y=1,x).'
                ' Need at least {}.'.format(hold_out_size + 1)
            )
        # construct the holdout set
        self.rng.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out, 0)
        y = np.delete(y, hold_out)
        # fit the inner estimator
        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_hold_out)[:, 1]

        # try:
        #     hold_out_predictions = hold_out_predictions[:, 1]
        # except TypeError:
        #     pass
        # update c, the positive proba estimate
        c = np.mean(hold_out_predictions)
        self.c = c
        self.estimator_fitted = True

    def predict(self, X, threshold=0.5):
        """Predict labels.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        threshold : float, default 0.5
            The decision threshold over probability to warrent a
            positive label.

        Returns
        -------
        y : array of int of shape = [n_samples]
            Predicted labels for the given inpurt samples.
        """
        raise NotImplementedError()


@run_all
def run(ao: [AbstractOpus, AbstractMultiOpus], redo=False):
    # if all(ao.has_score_type(m, ps.COMPETITOR) for m in ao.metric_methods_dict):
    #     return
    print(f'Computing competitor scores for {ao}')

    columns = [f'{ps.BASIC_PU}_{ao.tt_string(p)}' for p in [True, False]]

    def generate_empty():
        return pd.DataFrame(columns=columns + ps.idx).set_index(ps.idx)

    if redo:
        scores = {m: generate_empty() for m in ao.metric_methods_dict}
    else:
        def load_or_empty(m):
            if ao.has_score_type(m, ps.COMPETITOR):
                return ao.load_score_type(m, ps.COMPETITOR).sort_index()
            else:
                return generate_empty()

        scores = {m: load_or_empty(m) for m in ao.metric_methods_dict}

    save_counter = 0

    def store_results():
        for m, s in scores.items():
            s.index.names = ps.idx
            ao.store_score_type(metric=m, score_type=ps.COMPETITOR, scores=s)

    # Base PU learning
    for experiment in ao.experiment_iterator():
        experiment: SingleOpus

        if experiment.strategy == ps.UPDATE:
            # PU learning needs the converted, so we only do this for static
            continue
        elif experiment.strategy == ps.CD:
            pass
        elif experiment.strategy == ps.STATIC:
            pass
        else:
            raise ValueError(experiment.strategy)

        if all([experiment.index in df.index for df in scores.values()]):
            continue

        res = {m: pd.Series(dtype=float, name=experiment.index) for m in ao.metric_methods_dict}

        for is_phase1 in [True, False]:

            # Set up PU learner
            inner_estimator = ao.model_constructor_dict[experiment.model_name]()
            outer_estimator = AdaptedElkanotoPuClassifier(inner_estimator, random_state=0)

            # Get data
            if is_phase1:
                data_object = experiment.data_reference.phase_1_data
            else:
                data_object = experiment.data_reference.phase_2_data
            y = data_object.converted

            if y.values.sum() <= 1:
                continue
            x = data_object.x
            # Train PU learner
            try:
                outer_estimator.fit(x.values, y.values)
            except TypeError as e:
                # I'm not sure what this is
                if '_new_learning_node() got an unexpected keyword argument \'is_active_node\'' in str(e):
                    continue
                else:
                    raise e

            # Compute proba
            test_set = data_object.non_converted_only
            if outer_estimator.c == 0:
                continue
            probabilities = outer_estimator.predict_proba(test_set.x.values)
            positive_probabilities = get_positive_probabilities(probabilities, inner_estimator)

            # Compute labels
            y_pred = positive_probabilities >= 0.5
            y_true = test_set.labels

            for metric, metric_method in ao.metric_methods_dict.items():
                res[metric].loc[f'{ps.BASIC_PU}_{ao.tt_string(is_phase1)}'] = \
                    metric_method(y_pred=y_pred, y_true=y_true)
        for metric in scores.keys():
            scores[metric] = pd.concat([scores[metric], res[metric].to_frame().T], axis=0)

        save_counter += 1

        if save_counter == 100:
            print('[Saving Checkpoint]')
            store_results()
            save_counter = 0

    store_results()


def verify(oe: AbstractMultiOpus):
    for opus in oe:
        print(f'Verifying 04C-Competitors for {opus}: ', end='', flush=True)
        for metric in opus.metric_methods_dict:
            if not opus.has_score_type(metric=metric, score_type=ps.COMPETITOR):
                print(f'Missing competitors for {metric}')
                return False
            else:
                df = opus.load_score_type(metric, ps.COMPETITOR)
                ix = opus.experiment_dataframe_index
                ix = ix[ix.get_level_values(ps.STRATEGY) == ps.STATIC]
                try:
                    df.loc[ix]
                except KeyError:
                    print(f'{metric}: missing experiments')
                    return False
        print('All good')
    return True


def reset(oe: AbstractMultiOpus):
    for opus in oe:
        for metric in opus.metric_methods_dict:
            opus.remove_score_type(metric=metric, score_type=ps.COMPETITOR)
