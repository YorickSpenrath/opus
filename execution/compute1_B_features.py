import itertools

import pandas as pd

from ..objects import AbstractOpus
from ..objects.multi_opus import AbstractMultiOpus
from ..ops import optimize_threshold
from ._decorators import run_all, verify_all


@run_all
def run(amo: [AbstractOpus, AbstractMultiOpus], redo=False):
    single_phase_metrics = [f'acp', f'ancp'] + \
                           [f'oth_{m}' for m in amo.metric_methods_dict] + \
                           list(amo.metric_methods_dict)

    def phase_metrics(p):
        return [f'{x}_{amo.tt_string(p)}' for x in single_phase_metrics]

    all_features = phase_metrics(1) + phase_metrics(2) + phase_metrics(3)

    print(f'Computing features {amo}')
    for e in amo.experiment_iterator():
        if redo or not amo.has_features(e):
            features = pd.Series(dtype=float, index=all_features)
        else:
            # No redo AND there are features already
            features = amo.load_features(e).reindex(all_features)

        for phase in [1, 2, 3]:
            # Check if this phase is done
            if not pd.isna(features[phase_metrics(phase)]).any():
                continue

            # If not, load relevant data
            probabilities = amo.load_probabilities(e, phase)
            if phase == 1:
                xy_test = e.data_reference.phase_1_data.train_test_split()[1]
            elif phase == 2:
                xy_test = e.data_reference.phase_2_data
            elif phase == 3:
                xy_test = e.data_reference.phase_2_data.train_test_split()[1]
            else:
                raise NotImplementedError()

            # Compute for each metric
            for metric, method in amo.metric_methods_dict.items():
                # Get probabilities and labels of non-converted (the entities that are tested)
                pr_nc = probabilities[~xy_test.converted]
                lab_nc = xy_test.labels[~xy_test.converted]

                # Compute Optimal Threshold OTH
                oth = f'oth_{metric}_{amo.tt_string(phase)}'
                if pd.isna(features[oth]):
                    features[oth] = optimize_threshold(pr_nc, lab_nc, method)

                # Compute Metric value for OTH
                val = f'{metric}_{amo.tt_string(phase)}'
                if pd.isna(features[val]):
                    predicted_lab = pr_nc > features[oth]
                    features[val] = method(y_pred=predicted_lab, y_true=lab_nc)

            # Compute Average Converted Probability ACP
            acp = f'acp_{amo.tt_string(phase)}'
            if pd.isna(features[acp]):
                features[acp] = probabilities[xy_test.converted].mean()

            # Compute Average Non-Converted Probability ANCP
            ancp = f'ancp_{amo.tt_string(phase)}'
            if pd.isna(features[ancp]):
                features[ancp] = probabilities[~xy_test.converted].mean()

        amo.store_features(features, e)


@verify_all('Features')
def verify(opus):
    phases = [1, 2, 3]
    feature_names = {f'{oth_}{metric}_phase{x}' for (oth_, metric, x) in
                     itertools.product(['oth_', ''], opus.metric_methods_dict.keys(), phases)}
    feature_names = {*feature_names,
                     *{f'a{n}cp_phase{x}' for (n, x) in itertools.product(['', 'n'], phases)}}

    print(str(opus))
    for e in opus.experiment_iterator():
        if not opus.has_features(e):
            return False
        else:
            features = opus.load_features(e).index
            if not feature_names.issubset(features):
                return False
    return True
