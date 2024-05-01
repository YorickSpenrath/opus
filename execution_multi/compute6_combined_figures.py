import pandas as pd

import opus.functions
from .. import strings as ps
from ..execution.compute5_figures import make, limits_dict, threshold_sorter, predicted
from ..objects import AbstractOpus
from ..objects.multi_opus import AbstractMultiOpus
from ..ops import fail_to


def compute_df(t: [int, None], m: str, opus_experiment: AbstractMultiOpus, mode='metric'):
    """

    Parameters
    ----------
    t: int or None
        Timestep for which to get the results. If None, get results aggregated per timestep over all OPUS
    m: str
        Metric to get
    opus_experiment: AbstractMultiOpus
        OpusExperiment that hosts the data
    mode: str
        Whether to show the metric values ('metric'), the thresholds ('threshold') or the fails (ps.FAIL)

    Returns
    -------
    df: pd.DataFrame
        Results
    ci: pd.DataFrame or None
        Confidence interval of results if t is None, otherwise None.
    """
    if mode == 'threshold':
        def load(ao: AbstractOpus):
            return ao.load_all_thresholds(m)
    elif mode in ['metric', ps.FAIL]:
        def load(ao: AbstractOpus):
            return ao.load_all_scores(m)
    else:
        raise NotImplementedError(mode)

    ls = [load(ao).reset_index().assign(data=ao.short_name).fillna(ps.FAIL) for ao in opus_experiment]
    combined_df = pd.concat(ls, axis=0)

    # Apply FAIL flag --------------------------------------------------------------------------------------------------
    if mode == ps.FAIL:
        def mapper(x):
            if pd.isna(x):
                return x
            elif x == ps.FAIL:
                return 1
            else:
                return 0

        # Skip the first column, because that is time
        combined_df.iloc[:, 1:] = combined_df.iloc[:, 1:].applymap(mapper)

    # Get specific timestep --------------------------------------------------------------------------------------------
    def get_ts(ts):
        return combined_df[combined_df[ps.TIMESTEP] == ts].set_index('data').drop(columns=ps.TIMESTEP)

    # Single TS or All TS averaged?
    if t is not None:
        # Single Timestep
        return get_ts(t), None
    else:
        # All Timesteps, so create row per timestep and compute mean + ci
        last = opus_experiment.list_of_opus[-1]
        res_mean = pd.DataFrame()
        res_ci = pd.DataFrame()

        for tx in last.all_timesteps:
            dft = get_ts(tx).applymap(fail_to(pd.NA))

            def is_learned(x):
                z = threshold_sorter(x)
                if z[0] == predicted:
                    return z[1]
                else:
                    return ''

            learned_mask = dft.columns.map(lambda x: is_learned(x))

            df_mean_ci = dft.loc[:, learned_mask == ''].apply(opus.functions.mean_ci_gb(na_handling='ignore'))
            mean = df_mean_ci.iloc[0]
            ci = df_mean_ci.iloc[1]

            for learner in last.tpm_model_dict:
                dft_learner = dft.loc[:, learned_mask == learner].to_numpy().reshape(-1, 1)
                m_, c_ = opus.functions.mean_ci(dft_learner, na_handling='ignore')
                mean.loc[f'{ps.OPUS}:{last.tpm_shorthand_dict[learner]}'] = m_
                ci.loc[f'{ps.OPUS}:{last.tpm_shorthand_dict[learner]}'] = c_

            res_mean = pd.concat([res_mean, mean.rename(tx).to_frame().T], axis=0)
            res_ci = pd.concat([res_ci, ci.rename(tx).to_frame().T], axis=0)

        if mode == ps.FAIL:
            res_ci = None

        return res_mean, res_ci


def compute_specific_figure(opus_experiment: AbstractMultiOpus,
                            t: [int, None],
                            metric: str,
                            div4: bool = False,
                            mode='metric',
                            **kwargs):
    """
    Compute a specific figure

    Parameters
    ----------
    opus_experiment: AbstractMultiOpus
        Where the data is stored
    t: int or None
        What timestep to show. If None, all timesteps are shown as average over SingleOpus. If int, all SingleOpus
        are shown for that timestep.
    metric: str
        Metric to show
    div4: bool
        Whether only to include timesteps divisible by 4
    mode: str
        What to plot. 'default' plots the metric values, 'threshold' plots the thresholds, 'fail' plots fails

    Other Parameters
    ----------------
    kwargs: passed directly to make_heatmap

    Returns
    -------

    """

    assert mode in ['metric', 'threshold', ps.FAIL]

    df, ci = compute_df(t, metric, opus_experiment, mode)

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
        if mode != 'metric':
            name += f'_{mode}'
    else:
        name = kwargs.pop('name')

    opus_experiment.export_figure(
        make(df=df,
             ao=opus_experiment.list_of_opus[-1],
             title=kwargs.pop('title', f'{metric} @ {t}'),
             ci=ci,
             add_highlights=True,
             limits=(0, 1) if mode in ['threshold', ps.FAIL] else limits_dict[metric],
             mode=mode,
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
