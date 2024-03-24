import itertools
from typing import List

import pandas as pd

from ..functions import set_compare
from ._decorators import run_all
from .. import constants
from .. import strings as ps
from ..execution.compute2_threshold_prediction_models import split_and_preprocess_tpm
from ..objects.abstract_opus import AbstractOpus
from ..objects.multi_opus import AbstractMultiOpus


def run(ao: [AbstractOpus, AbstractMultiOpus], list_of_oa: List[AbstractOpus] = None, redo=False):
    if isinstance(ao, AbstractMultiOpus):
        assert list_of_oa is None
        for opus in ao:
            run(opus, list_of_oa=ao.get_cross(opus), redo=redo)
        return

    print(f'Computing adapted thresholds for {ao}')
    tab = ao.load_tpm_df()
    for metric in ao.metric_methods_dict.keys():

        print(f'\t{metric}')

        if not redo and ao.has_thresholds(metric):
            res = ao.load_thresholds(metric)
        else:
            res = pd.DataFrame(index=tab.index)

        for method, compute in ao.default_thresholds_dict.items():
            if method in res.columns:
                # Skip if done
                continue
            res[method] = compute(tab, metric)

        if list_of_oa is not None:
            for other_ao in list_of_oa:
                if other_ao == ao:
                    continue
                else:
                    for tpm_model_name in other_ao.tpm_model_dict.keys():
                        name = f'{str(other_ao)}:{tpm_model_name}'
                        if name in res.columns:
                            # Skip if done
                            continue
                        model = other_ao.load_tpm(tpm_model_name, metric)
                        ohe = other_ao.load_tpm_ohe()
                        x, _ = split_and_preprocess_tpm(ao, metric, ohe)
                        other_x, _ = split_and_preprocess_tpm(other_ao, metric)
                        x = x.reindex(columns=other_x.columns, fill_value=0)
                        mask = ~x.isna().any(axis=1)
                        data = model.predict(x[mask])
                        res[name] = pd.Series(data=data, index=tab.index[mask]).reindex(tab.index)

        ao.store_thresholds(res, metric)


@run_all
def reset(opus: AbstractOpus):
    for metric in opus.metric_methods_dict.keys():
        opus.remove_thresholds(metric)


def verify_single_base(opus: AbstractOpus, opus_experiment: AbstractMultiOpus, threshold: bool):
    if threshold:
        n = '03-Thresholds'

    else:
        n = '04A-OPUS scores'

    print(f'Verifying {n} for {opus}: ', end='', flush=True)

    learned_methods = [f'{other}:{base_tpm}' for (other, base_tpm) in
                       itertools.product(opus_experiment.get_cross(opus), opus.tpm_model_dict.keys())]

    for metric in opus.metric_methods_dict:

        if threshold:
            if not opus.has_thresholds(metric):
                print(f'missing for {metric}')
                return False
            else:
                df = opus.load_thresholds(metric)
        else:
            if not opus.has_score_type(metric, ps.OPUS):
                print(f'missing for {metric}')
                return False
            else:
                df = opus.load_score_type(metric, ps.OPUS)

        # Columns
        expected = set(
            constants.baseline_methods
            + constants.heuristic_methods
            + constants.comparison_methods
            + learned_methods)

        missing, superfluous, okay = set_compare(df.columns, expected)
        if not okay:
            print(f'{metric}: missing {missing}, superfluous {superfluous}')
            return False

        try:
            df.loc[opus.experiment_dataframe_index]
        except KeyError:
            print(f'{metric}: missing experiments')
            return False

        # TODO you are not checking for duplicates..

    print('All good')
    return True


def verify_single(opus: AbstractOpus, opus_experiment: AbstractMultiOpus):
    return verify_single_base(opus, opus_experiment, threshold=True)


def verify(opus_experiment: AbstractMultiOpus):
    return all(map(lambda x: verify_single(x, opus_experiment), opus_experiment))
