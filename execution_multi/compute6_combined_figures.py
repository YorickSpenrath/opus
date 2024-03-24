import pandas as pd

import opus.functions
from ..functions import mean_ci, mean_ci_gb
from .. import strings as ps
from ..execution.compute5_figures import make, limits_dict, threshold_sorter, predicted
from ..objects.multi_opus import AbstractMultiOpus


def compute_df(t: [int, None], m: str, opus_experiment: AbstractMultiOpus):
    """

    Parameters
    ----------
    t: int or None
        Timestep for which to get the results. If None, get results aggregated per timestep over all OPUS
    m: str
        Metric to get
    opus_experiment: AbstractMultiOpus
        OpusExperiment that hosts the data

    Returns
    -------
    df: pd.DataFrame
        Results
    ci: pd.DataFrame or None
        Confidence interval of results if t is None, otherwise None.
    """
    ls = [ao.load_all_scores(m).reset_index().assign(data=ao.short_name) for ao in opus_experiment]
    combined_df = pd.concat(ls, axis=0)

    def get_ts(ts):
        return combined_df[combined_df[ps.TIMESTEP] == ts].set_index('data').drop(columns=ps.TIMESTEP)

    if t is not None:
        # Single Timestep
        return get_ts(t), None
    else:
        last = opus_experiment.list_of_opus[-1]
        res_mean = pd.DataFrame()
        res_ci = pd.DataFrame()

        for tx in last.all_timesteps:
            dft = get_ts(tx)

            def is_learned(x):
                z = threshold_sorter(x)
                if z[0] == predicted:
                    return z[1]
                else:
                    return ''

            learned_mask = dft.columns.map(lambda x: is_learned(x))

            mean_ci = dft.loc[:, learned_mask == ''].apply(opus.functions.mean_ci_gb(na_handling='ignore'))
            mean = mean_ci.iloc[0]
            ci = mean_ci.iloc[1]

            for learner in last.tpm_model_dict:
                m_, c_ = opus.functions.mean_ci(dft.loc[:, learned_mask == learner].to_numpy().reshape(-1, 1),
                                                na_handling='ignore')
                mean.loc[f'{ps.OPUS}:{last.tpm_shorthand_dict[learner]}'] = m_
                ci.loc[f'{ps.OPUS}:{last.tpm_shorthand_dict[learner]}'] = c_

            res_mean = pd.concat([res_mean, mean.rename(tx).to_frame().T], axis=0)
            res_ci = pd.concat([res_ci, ci.rename(tx).to_frame().T], axis=0)

        return res_mean, res_ci


def compute_specific_figure(opus_experiment: AbstractMultiOpus,
                            t: [int, None],
                            metric: str,
                            div4: bool = False,
                            **kwargs):
    df, ci = compute_df(t, metric, opus_experiment)

    if div4 and (t is None):
        df = df[df.index % 4 == 0]
        if ci is not None:
            ci = ci[ci.index % 4 == 0]

    if 'name' not in kwargs:
        name = metric
        if t is not None:
            name += f'_t{t}'
        if div4:
            name += f'_div4'
        if kwargs.get('keep_tem_labels', True):
            name += '_labels'
        if kwargs.get('add_arrows', False):
            name += '_arrows'
    else:
        name = kwargs.pop('name')

    opus_experiment.export_figure(
        make(df=df,
             ao=opus_experiment.list_of_opus[-1],
             title=kwargs.pop('title', f'{metric} @ {t}'),
             ci=ci,
             add_highlights=True,
             limits=limits_dict[metric],
             **kwargs)[0],
        name=name)


def run(opus_experiment: AbstractMultiOpus):
    list_of_abstract_opus = opus_experiment.list_of_opus

    def out(figure, x):
        opus_experiment.export_figure(figure, x)

    last = list_of_abstract_opus[-1]
    for m in last.metric_methods_dict:

        for t in last.all_timesteps:
            dft = compute_df(t, m, opus_experiment)[0]
            f, ax = make(dft, last, f'{m}@{t}', limits=limits_dict[m], add_highlights=True)
            out(f, f'{m}-{t}')

        res_mean, res_ci = compute_df(None, m, opus_experiment)

        out(make(res_mean, last, m, limits=limits_dict[m],
                 add_highlights=True,
                 ci=res_ci)[0], f'{m}-combined')
