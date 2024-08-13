import numpy as np
import pandas as pd
from pulearn import ElkanotoPuClassifier
from ._decorators import run_all
from ..execution.compute1_A_probabilities import get_positive_probabilities
from ..objects import AbstractOpus
from .. import strings as ps
from ..objects.multi_opus import AbstractMultiOpus
from ..objects.single_opus import SingleOpus
from .competitors import xgb_pu


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

    columns = [f'{ps.BASIC_PU}_{ao.tt_string(p)}' for p in [True, False]] + \
              [f'{ps.UPU}_{ao.tt_string(p)}' for p in [True, False]] + \
              [f'{ps.NNPU}_{ao.tt_string(p)}' for p in [True, False]]

    def generate_empty():
        return pd.DataFrame(columns=columns + ps.idx).set_index(ps.idx)

    if redo:
        scores = {m: generate_empty() for m in ao.metric_methods_dict}
    else:
        def load_or_empty(m):
            if ao.has_score_type(m, ps.COMPETITOR):
                return ao.load_score_type(m, ps.COMPETITOR).sort_index().reindex(columns=columns)
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

        def verify_metric(df_metric):
            if experiment.index not in df_metric.index:
                return False
            if df_metric.loc[experiment.index].isna().any():
                return False
            return True

        if all(verify_metric(df_metric) for df_metric in scores.values()):
            continue
        for is_phase1 in [True, False]:

            def log(msg):
                for m in ao.metric_methods_dict.keys():
                    original_message = scores[m].loc[experiment.index, f'Note_{ao.tt_string(is_phase1)}']
                    if pd.isna(original_message):
                        new_message = msg
                    else:
                        new_message = original_message + ',' + msg
                    scores[m].loc[experiment.index, f'Note_{ao.tt_string(is_phase1)}'] = new_message
                    scores[m].loc[experiment.index, f'{ps.BASIC_PU}_{ao.tt_string(is_phase1)}'] = ps.FAIL

            # BASIC PU -------------------------------------------------------------------------------------------------
            def check_if_pu_is_done(df):
                if experiment.index not in df.index:
                    return False
                return not pd.isna(df.loc[experiment.index, f'{ps.BASIC_PU}_{ao.tt_string(is_phase1)}'])

            do_pu = (not all(map(check_if_pu_is_done, scores.values()))) & (experiment.strategy != ps.UPDATE)

            if do_pu:

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
                    log(f'{y.values.sum()} positives')
                    continue
                x = data_object.x
                # Train PU learner
                try:
                    outer_estimator.fit(x.values, y.values)
                except TypeError as e:
                    # I'm not sure what this is
                    if '_new_learning_node() got an unexpected keyword argument \'is_active_node\'' in str(e):
                        log('AHOT ERROR')
                        continue
                    else:
                        raise e

                # Compute proba
                test_set = data_object.non_converted_only
                for m in ao.metric_methods_dict.keys():
                    scores[m].loc[experiment.index, 'c'] = outer_estimator.c

                if outer_estimator.c == 0:
                    log('c=0')
                    continue
                probabilities = outer_estimator.predict_proba(test_set.x.values)
                positive_probabilities = get_positive_probabilities(probabilities, inner_estimator)

                # Compute labels
                y_pred = positive_probabilities >= 0.5
                y_true = test_set.labels

                for metric, metric_method in ao.metric_methods_dict.items():
                    scores[metric].loc[experiment.index, f'{ps.BASIC_PU}_{ao.tt_string(is_phase1)}'] = \
                        metric_method(y_pred=y_pred, y_true=y_true)

            # UPU / NNPU -----------------------------------------------------------------------------------------------
            for pu in [ps.UPU, ps.NNPU]:
                def check_if_done(df):
                    if experiment.index not in df.index:
                        return False
                    return not pd.isna(df.loc[experiment.index, f'{pu}_{ao.tt_string(is_phase1)}'])

                do_pu = (not all(map(check_if_done, scores.values()))) & \
                        (experiment.model_name == ps.DECISION_TREE)

                if do_pu:
                    if is_phase1:
                        train_data, test_data = experiment.data_reference.phase_1_data.train_test_split()
                    else:

                        def check_is_highest(m):
                            all_scores = scores[m].loc[experiment.ts].loc[:, f'{pu}_{ao.tt_string(True)}']

                            # If no (valid) scores yet: there is no point in P2
                            if pd.isna(all_scores).all():
                                return False

                            # Find the best score, and check if _this_ experiment has the best score in p1
                            # If so, we compute the second phase results. Otherwise we won't use it anyway
                            # TODO: also do this for all other experiments?
                            best_score = all_scores.max()
                            score_p1 = scores[m].loc[experiment.index, f'{pu}_{ao.tt_string(True)}']
                            return score_p1 == best_score

                        if not any(check_is_highest(m) for m in ao.metric_methods_dict.keys()):
                            continue

                        train_data = experiment.data_reference.phase_1_data
                        test_data = experiment.data_reference.phase_2_data

                    # Compute label frequency
                    phase1_data = experiment.data_reference.phase_1_data
                    label_frequency = (phase1_data.converted.sum()) / (phase1_data.labels.sum())

                    if experiment.strategy == ps.STATIC:
                        pass
                    elif experiment.strategy == ps.UPDATE:
                        # Add new positive points to training data
                        train_data = train_data + test_data.converted_only
                    else:
                        raise NotImplementedError()

                    # Keep only non_converted from test data
                    test_data = test_data.non_converted_only

                    # Match Features
                    test_data = test_data.match_features(train_data)

                    # Get predicted labels -----------------------------------------------------------------------------
                    y_pred = xgb_pu.PUBoost(obj=pu, random_state=0, label_freq=label_frequency) \
                                 .fit(train_data.x, train_data.labels) \
                                 .inplace_predict(test_data.x) >= 0.50

                    # Compute and save the metric scores ---------------------------------------------------------------
                    for metric, metric_method in ao.metric_methods_dict.items():
                        scores[metric].loc[experiment.index, f'{pu}_{ao.tt_string(is_phase1)}'] = \
                            metric_method(y_pred=y_pred, y_true=test_data.labels)

            # # PU_HDT ---------------------------------------------------------------------------------------------------
            # def check_if_pu_hdt_is_done(df):
            #     return not pd.isna(df.loc[experiment.index, f'{ps.PU_HDT}_{ao.tt_string(is_phase1)}'])
            #
            # if not all(map(check_if_pu_hdt_is_done, scores.values())):
            #
            #     if is_phase1:
            #         # Train PH-HDT on the labelled
            #         xy_train, xy_validation = experiment.data_reference.phase_1_data.train_test_split()
            #     else:
            #
            #         def check_is_highest(m):
            #             all_scores = scores[m].loc[:, f'{ps.PU_HDT}_{ao.tt_string(True)}']
            #
            #             # If no (valid) scores yet: there is no point in P2
            #             if pd.isna(all_scores).all():
            #                 return False
            #
            #             # Find the best score, and check if _this_ experiment has the best score
            #             # If so, we compute the second phase results. Otherwise we skip it, as we won't use it anyway
            #             # TODO: also do this for all other experiments?
            #             best_score = all_scores.max()
            #             score_p1 = scores[m].loc[experiment.index, f'{ps.PU_HDT}_{ao.tt_string(True)}']
            #             return score_p1 == best_score
            #
            #         # Pre-emptive check if the P1 is (one of) the highest. If not for any metric
            #         if not any(check_is_highest(m) for m in ao.metric_methods_dict.keys()):
            #             continue
            #
            #         xy_train = experiment.data_reference.phase_1_data
            #         xy_validation = experiment.data_reference.phase_2_data
            #
            #     x_train = xy_train.x
            #     y_train = xy_train.converted
            #
            #     # train model
            #     model = pu_tree_simplified.PuHdt(random_state=0, max_depth=5)
            #     model.fit(x_train, y_train, p_y=np.mean(y_train))
            #     probabilities = model.predict_proba(xy_validation.x)
            #     positive_probabilities = get_positive_probabilities(probabilities, model)
            #
            #     for metric, metric_method in ao.metric_methods_dict.items():
            #         # Compute (p1) or get (p2) the OTH
            #         if is_phase1:
            #             oth = find_optimal_threshold(y_test=xy_validation.labels, y_prob=positive_probabilities,
            #                                          metric_method=metric_method)
            #             scores[metric].loc[experiment.index, f'{ps.PU_HDT}_{ao.tt_string(True)}_oth'] = oth
            #         else:
            #             oth = scores[metric].loc[experiment.index, f'{ps.PU_HDT}_{ao.tt_string(True)}_oth']
            #
            #         # Assign score based on OTH
            #         score = metric_method(y_true=xy_validation.labels, y_test=positive_probabilities >= oth)
            #         scores[metric].loc[experiment.index, f'{ps.PU_HDT}_{ao.tt_string(is_phase1)}'] = score

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
                if df.isna().any().any():
                    print(f'{metric}: missing values')

        print('All good')
    return True


def reset(oe: AbstractMultiOpus):
    for opus in oe:
        for metric in opus.metric_methods_dict:
            opus.remove_score_type(metric=metric, score_type=ps.COMPETITOR)
