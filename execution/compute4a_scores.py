import pandas as pd

from ._decorators import run_all
from .compute3_adapted_thresholds import verify_single_base
from ..objects.multi_opus import AbstractMultiOpus
from ..objects.single_opus import SingleOpus
from ..objects import AbstractOpus, DataObject
from .. import strings as ps


@run_all
def run(ao: [AbstractOpus, AbstractMultiOpus], redo=False):
    print(f'Computing scores for {ao}')
    if not redo:

        def load_or_empty(m):
            if ao.has_score_type(m, ps.OPUS):
                return ao.load_score_type(m, ps.OPUS).sort_index()
            else:
                return pd.DataFrame(columns=ps.idx).set_index(ps.idx)

        scores = {m: load_or_empty(m) for m in ao.metric_methods_dict.keys()}
    else:
        scores = {m: pd.DataFrame(columns=ps.idx).set_index(ps.idx) for m in ao.metric_methods_dict.keys()}

    save_counter = 0
    thresholds = {m: ao.load_thresholds(m) for m in ao.metric_methods_dict.keys()}

    def store_results():
        for m, s in scores.items():
            s.index.names = ps.idx
            ao.store_score_type(metric=m, score_type=ps.OPUS, scores=s)

    for experiment in ao.experiment_iterator():
        experiment: SingleOpus

        if all([experiment.index in df.index for df in scores.values()]):
            continue

        p2: DataObject = experiment.data_reference.phase_2_data.non_converted_only
        pr = ao.load_probabilities(experiment, False).reindex(p2.index)
        y_true = p2.labels

        for metric, metric_method in ao.metric_methods_dict.items():
            res = pd.Series(dtype=float, name=experiment.index)
            for th_name, th_value in thresholds[metric].loc[experiment.index].items():
                if pd.isna(th_value):
                    res.loc[th_name] = pd.NA
                else:
                    y_pred = pr >= th_value
                    res.loc[th_name] = metric_method(y_true=y_true, y_pred=y_pred)
            scores[metric] = pd.concat([scores[metric], res.to_frame().T], axis=0).sort_index()

        save_counter += 1
        if save_counter == 100:
            print('[Saving checkpoint]')
            store_results()
            save_counter = 0

    store_results()


def verify_single(opus: AbstractOpus, opus_experiment: AbstractMultiOpus):
    return verify_single_base(opus, opus_experiment, threshold=False)


def verify(opus_experiment: AbstractMultiOpus):
    return all(map(lambda x: verify_single(x, opus_experiment), opus_experiment))
