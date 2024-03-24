import warnings

import matplotlib.pyplot as plt
import pandas as pd

from ..functions import make_heatmap
from ._decorators import run_all
from .. import strings as ps
from ..constants import heuristic_methods, baseline_methods
from ..objects.abstract_opus import AbstractOpus

# Limits for the heatmaps
limits_dict = {ps.ACCURACY: (0, 1), ps.F1: (0, 1), ps.NAPE_CONVERTED_OBJECT_COUNT: (-2, 0)}

# Sorting
baseline = 0
heuristic = 1
predicted = 2
compare = 3
competitor = 4
alternative = 5

# Names for arrows on top
names = {baseline: 'Baseline', heuristic: 'Heuristic', predicted: '$TPM$', compare: 'Comparison',
         competitor: 'Competitor', alternative: 'Alternative'}


def threshold_sorter(c):
    """
    Sort Threshold Estimation Methods (TEMs)

    Parameters
    ----------
    c: str
        raw TEM name

    Returns
    -------
    key: Tuple
        Sorting key

    """
    if c in baseline_methods:
        return baseline, baseline_methods.index(c)
    elif c == 'previous':
        return baseline, len(baseline_methods)
    elif c in heuristic_methods:
        return heuristic, heuristic_methods.index(c)
    elif c == 'optimal':
        return compare, 1
    elif c == 'mean-phase1':
        return heuristic, len(heuristic_methods) + 1
    elif c == 'mean-phase2':
        return compare, 0
    elif c == 'retrain':
        return compare, 2
    elif ':' in c and c.count(':') == 1:
        a, b = c.split(':')
        if a == ps.ALTERNATIVE:
            return alternative, b
        elif a == ps.COMPETITOR:
            return competitor, b
        else:
            return predicted, b, a
    else:
        raise NotImplementedError(c)


def make(df: pd.DataFrame,
         ao: AbstractOpus,
         title: str,
         ci: [pd.DataFrame, None] = None,
         add_highlights: bool = False,
         add_arrows: bool = False,
         tem_filter: [None, str, callable] = 'default',
         keep_tem_labels: bool = True, **kwargs):
    """

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with results
    ao: AbstractOpus
        AbstractOpus to use for naming etc.
    title: str or bool
        Title of heatmap. Ignored in add_arrows is True. If False, no title is added.
    ci: pd.DataFrame or None
        If not None, passed as confidence intervals to make_heatmap
    add_highlights: bool
        If True, boldface values higher than regular TEM and italize values lower than regular TEM
    add_arrows: bool
        If True, group TEMs with an arrow above them and remove title. If False, show title
    tem_filter: None, callable or str
        If 'default': remove empty TEMs and sanity-check TEMs. If None, remove no TEMs. If callable, apply given
        tem_filter to df and ci.
    keep_tem_labels: bool
        If True, show TEM labels on horizontal. If False, skip.

    Other Parameters
    ----------------
    Passed directly into make_heatmap

    Returns
    -------
    f : plt.Figure
        Resulting Figure
    ax : plt.Axes
        Resulting Axes
    """

    # Filter out threshold estimation methods ==========================================================================
    if tem_filter == 'default':
        def tem_filter(df_):
            df_ = df_.copy().loc[:, ~df_.isna().all(axis=0)]
            df_ = df_[[c for c in df_.columns if c not in ['mean-phase1', 'mean-phase2', '1', '0']]]
            return df_
    elif tem_filter is None:
        def tem_filter(df_):
            return df_
    elif callable(tem_filter):
        pass
    else:
        raise NotImplementedError(f'Not implemented tem_filter={tem_filter}')

    df = tem_filter(df)
    if ci is not None:
        ci = tem_filter(ci)

    # Compute number of each TEM type ==================================================================================
    number_of_each = df.columns.map(lambda x: threshold_sorter(x)[0]).value_counts().sort_index().reindex(
        range(len(names)), fill_value=0)

    # Rename and sort TEMs =============================================================================================
    def translate_methods(x):
        if x.startswith(f'{ps.OPUS}:'):
            return x[len(ps.OPUS) + 1:]
        if x.count(':') == 1 and x.split(':')[0] not in [ps.ALTERNATIVE, ps.COMPETITOR]:
            return ao.translate_tpm(x)
        return ao.threshold_translations.get(x, x)

    def prepare_df(df_):
        df_ = df_.copy()
        df_ = df_[sorted(df_.columns, key=threshold_sorter)]
        df_ = df_.rename(columns=translate_methods)
        return df_

    df = prepare_df(df)
    if ci is not None:
        ci = prepare_df(ci)

    # Highlights for higher/better than previous =======================================================================
    if add_highlights:
        df_bold = df.gt(df[translate_methods('previous')], axis=0)
        df_italic = df.lt(df[translate_methods('previous')], axis=0)
    else:
        df_bold = None
        df_italic = None

    # Plot figure ======================================================================================================
    f, ax = make_heatmap(
        data=df.rename(columns=translate_methods),
        x_rotation=90,
        na_color='w',
        df_ci=ci,
        num_dec=2,
        value_font_size=6,
        aspect='auto' if ci is None else .35,
        ci_on_new_line=False,
        boldface_mask=df_bold,
        italic_mask=df_italic,
        # TODO: add metric variable and set limit here
        **kwargs)

    # Remove horizontal labels
    if keep_tem_labels:
        pass
    else:
        ax.set_xticks([])

    # Add vertical lines between each set of threshold estimation methods ==============================================
    for th_type_index, z in number_of_each.cumsum().iloc[:-1].items():
        if number_of_each[th_type_index + 1] > 0:
            ax.axvline(x=z - 0.5, color='k', lw=1)
            ax.axvline(x=z - 0.5, color='w', ls=':', lw=1)

    # Fix the time/dataset labels (I'm not sure anymore why) ===========================================================
    ticks = []
    tick_labels = []
    for i, j in enumerate(df.index):
        if j != '':
            ticks.append(i)
            tick_labels.append(j)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # Add arrows on top of the heatmap to group threshold estimation methods ===========================================
    if add_arrows:
        left = number_of_each.cumsum() - number_of_each - 0.5
        right = number_of_each.cumsum() - 0.5
        short_names = {baseline: 'BL'}

        for i in names.keys():
            x_left = left[i]
            x_right = right[i]
            y = 1.1
            if number_of_each[i] < 2:
                text = False
                arrow = False
            else:
                text = True
                arrow = True

            if ci is None:
                if number_of_each[i] < 4:
                    s = short_names.get(i, '')
                    if i not in short_names:
                        warnings.warn(f'Cannot find short name for {i}')
                        arrow = False
                        text = False
                else:
                    s = names[i]
            else:
                s = names[i]

            if arrow:
                ax.annotate('',
                            xy=((x_left + 0.5) / len(df.columns), y), xycoords=ax.transAxes,
                            xytext=((x_right + 0.5) / len(df.columns), y), textcoords=ax.transAxes,
                            arrowprops=dict(arrowstyle='<|-|>, head_width=0.1', color=(0.25, 0.25, 0.25), lw=0.5))

            if text:
                ax.text(x=(x_left + x_right + 1) / 2 / len(df.columns), y=y, s=s, transform=ax.transAxes,
                        va='center', ha='center',
                        fontdict=dict(size=(6 if ci is not None else 6)),
                        bbox=dict(facecolor='w', ec='none', pad=0))
    # Or alternatively set the title
    else:
        if title is not False:
            ax.set_title(title)

    # I'm not sure why this is here... =================================================================================
    w, h = f.get_size_inches()
    f.set_size_inches(w, h)

    return f, ax


@run_all
def run(ao: AbstractOpus):
    for m in ao.metric_methods_dict.keys():
        df, best_hp_previous = ao.load_all_scores(m, True)
        f, ax = make(df, ao, m, limits=limits_dict[m], add_highlights=True)
        ao.store_figure(f, m)

        # Thresholds
        thresholds = ao.load_thresholds(m)
        df = thresholds.loc[best_hp_previous].reset_index(level=[1, 2], drop=True)
        f, ax = make(df, ao, f'th-{m}')
        ao.store_figure(f, f'th-{m}')
