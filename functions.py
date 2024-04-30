import warnings
from typing import Tuple, Set, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.colors import ListedColormap
from scipy import stats


def set_compare(actual_set, expected_set) -> Tuple[Set, Set, bool]:
    missing = set(expected_set).difference(actual_set)
    superfluous = set(actual_set).difference(expected_set)
    okay = (not missing) and (not superfluous)

    return missing, superfluous, okay


def make_heatmap(data: pd.DataFrame,
                 inverted: bool = False,
                 num_dec: [List[int], int] = 2,
                 skip_zero: [List[bool], bool] = True,
                 ax: [None, plt.Axes] = None,
                 df_ci: [pd.DataFrame, None] = None,
                 show_labels: [str, bool] = True,
                 show_values: [str, bool] = True,
                 value_font_size: int = 4,
                 aspect: [str, float] = 'auto',
                 label_font_size: int = 6,
                 x_rotation: int = 0,
                 cmap=cm.get_cmap('inferno', 1000),
                 na_color: str = 'k',
                 limits: [Tuple[float, float], str, None] = None,
                 boldface_mask: [pd.DataFrame, None] = None,
                 italic_mask: [pd.DataFrame, None] = None,
                 ci_on_new_line: bool = True,
                 text_values: [None, pd.DataFrame] = None):
    """
    Makes a heatmap.

    Parameters
    ----------
    data: pd.DataFrame
        The data that is used for the values
    inverted: bool
        If True, the values of da. If bool and False, the data values are used for colours, but
        inverted
    num_dec: int or iterable of int
        Number of decimals in which the data is shown. If iterable, one value per row
    skip_zero: bool or iterable of bool
        Whether to skip the "0" in a "0.xx" value. If iterable, one value per row
    ax: plt.Axes or None
        If not None, the heatmap will be plot in this Axes. Otherwise a new Axes is created
    df_ci: pd.DataFrame
        DataFrame with confidence interval values
    show_labels: bool or str
        If True, labels are shown. If False, no labels are shown. If 'x', only x (column) labels are shown. If 'y', only
         y (index) labels are show
    show_values: bool or str
        If True: values are shown.
        If False: no values are shown.
        If 'corners': corner values are shown.
        If 'center': corner values are shown.
        If tuple of str: combination the above values.
    value_font_size: int
        The fontsize of the text in the cell
    label_font_size: int
        The fontsize of the text on the axislabels
    aspect: [str, float]
        The aspect parameter passed to the (created) ax. If str and 'auto': determine based on whether df_ci is given
    x_rotation: int
        Rotation of the x labels.
    cmap: ColorMap
        ColorMap to use. Default is inferno
    na_color: str or tuple
        The colour for NA values.
    limits: Tuple[float] or None or str
        The limits to use for data colouring. If None, the max/min value are taken. If not None, the data is clipped. If
        str and 'row', the limits (and as such colours) are scaled per row. If str and 'column', the limits (and as such
         colours) are scaled per columns.
    italic_mask: pd.DataFrame or None
        Whether a value should be formatted in italic or not. If None, no values are formatted in italic
    boldface_mask: pd.DataFrame or None
        Whether a value should be formatted in bold or not. If None, no values are formatted in bold
    ci_on_new_line: bool
        Whether the CI value is added to the next line or shown on the same line. Ignored if CI is not given
    text_values: None or pd.DataFrame
        If not None, the text added for the values is taken from this. Otherwise the data values (and CI) are used.

    Returns
    -------
    f: plt.Figure
        if ax is None, a new Figure. If ax is not None, this is skipped
    ax: plt.Axes
        if ax is None, a new Axes. Otherwise, ax

    """
    # TODO: estimate value/label font + aspect from data/text
    # TODO: allow colour clipping i.e. first clip the values before computing the colours
    #  (as alternative to having data_for_colours)

    # Take care of NA values ===========================================================================================
    na_mask = data.isna()

    # For all dataframe entries, ensure that the have the same index and columns as the data
    def assert_df(df):
        if df is not None:
            assert (df.index == data.index).all()
            assert (df.columns == data.columns).all()

    assert_df(df_ci)
    assert_df(italic_mask)
    assert_df(boldface_mask)
    assert_df(text_values)

    if df_ci is not None and text_values is not None:
        warnings.warn('Warning: df_ci and text_values are both given, df_ci is ignored')

    # Fill value =======================================================================================================
    data = data.fillna(data.mean().mean())

    # Data for colours =================================================================================================
    def normalize_data(df, bot, top):
        if bot == top:
            foo = np.ones_like(df) * 0.5
        else:
            foo = (df.clip(lower=bot, upper=top).to_numpy() - bot) / (top - bot)

        return pd.DataFrame(index=df.index, columns=df.columns, data=foo)

    def compute_data_for_colours(dfx, lim):
        if lim is None:
            # No limit, so normalize with actual data values
            dfc = normalize_data(dfx, dfx.min().min(), dfx.max().max())
        elif isinstance(lim, str) and lim in ['row', 'column']:
            # Scale per row or per column
            if lim == 'row':
                axis = 1
            elif lim == 'column':
                axis = 0
            else:
                raise NotImplementedError(f'Invalid string value for limits: {lim}')
            dfc = dfx.apply(lambda sr: (sr - min(sr)) / (max(sr) - min(sr)), axis=axis).to_numpy()
        elif isinstance(lim, tuple) and len(lim) == 2:
            # Assuming floats, scale with given values
            assert len(lim) == 2, f'Must give 2 limits : {lim}'
            dfc = normalize_data(dfx, *map(float, lim))
        elif isinstance(lim, List) and len(lim) == len(dfx):
            dfc = pd.DataFrame()
            for i in range(len(dfx)):
                new_row = compute_data_for_colours(dfx.iloc[i:i + 1], limits[i])[0]
                new_sr = pd.Series(new_row, index=dfx.columns, name=dfx.index[i])
                dfc = pd.concat([dfc, new_sr.to_frame().T], axis=0)
        else:
            raise NotImplementedError(f'Given limits {lim} not recognized')

        return dfc

    data_for_colours = compute_data_for_colours(data, limits)

    if inverted:
        data_for_colours = 1 - data_for_colours

    # Extend skip_zero to rows if necessary ============================================================================
    if isinstance(skip_zero, bool):
        skip_zero = [skip_zero] * len(data)

    # Extend num_dec to rows if necessary ==============================================================================
    if isinstance(num_dec, int):
        num_dec = [num_dec] * len(data)

    # Create figure if necessary =======================================================================================
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = None

    # Heatmap ==========================================================================================================
    # Scaling is determined above
    ax.imshow(data_for_colours, cmap=cmap, vmin=0, vmax=1)

    if text_values is None:
        text_values = pd.DataFrame(columns=data.columns, index=data.index)
        for x in range(len(data.columns)):
            for y, skip_zero_for_row, num_dec_for_row in zip(range(len(data.index)), skip_zero, num_dec):
                def fmt(v):
                    if isinstance(v, str):
                        return v
                    s = f'{v:.{num_dec_for_row}f}'
                    if skip_zero_for_row and abs(v) < 1:
                        s = s.replace('0.', '.')
                    return s

                txt = fmt(data.iloc[y, x])
                if df_ci is not None:
                    txt += ('\n' if ci_on_new_line else ' ')
                    txt += rf'$\pm$' + fmt(df_ci.iloc[y, x])

                text_values.iloc[y, x] = txt

    # Determine which values to show ===================================================================================
    if isinstance(show_values, bool):
        def should_show(x_, y_):
            return show_values
    else:
        # These are special values, such as corners, center etc.

        # Verify that it is a single or multiple correct strings
        correct_values = ['inner_corner', 'corner', 'center']
        if isinstance(show_values, str):
            # Single value, assert it is a correct one
            assert show_values in correct_values, f'invalid value : {show_values}'
            show_values = [show_values]
        else:
            # Multiple values, assert they are all strings and all valid
            show_values = set(show_values)
            assert all([isinstance(i, str) for i in show_values]), 'Multiple show_values argument should all be string'
            incorrect = show_values.difference(correct_values)
            assert len(incorrect) == 0, f'Multiple show_values, but found incorrect: {incorrect}'

        show_corners = 'corner' in show_values
        show_centers = 'center' in show_values
        show_inner_corners = 'inner_corner' in show_values

        def should_show(x_, y_):
            if show_corners and x_ in [0, len(data.columns) - 1] and y_ in [0, len(data.index) - 1]:
                return True
            if show_centers and x_ == (len(data.columns) - 1) // 2 and y_ == (len(data.index) - 1) // 2:
                return True
            if show_inner_corners and x_ in [1, len(data.columns) - 2] and y_ in [1, len(data.index) - 2]:
                return True
            return False

    # Show all the necessary values ====================================================================================
    if show_values:  # skip all values if show_values is False to optimize runtime
        for x in range(len(text_values.columns)):
            for y, skip_zero_for_row, num_dec_for_row in zip(range(len(text_values.index)), skip_zero, num_dec):
                if not should_show(x, y) or pd.isna(text_values.iloc[y, x]):
                    continue

                # White in the bottom, black in the top
                col = 'w' if (data_for_colours.iloc[y, x] < 0.5) else 'k'

                # TODO: na text?
                if not na_mask.iloc[y, x]:
                    # Format number of decimals
                    t = text_values.iloc[y, x]

                    kw = dict()
                    if italic_mask is not None and italic_mask.iloc[y, x]:
                        kw['style'] = 'italic'

                    if boldface_mask is not None and boldface_mask.iloc[y, x]:
                        kw['weight'] = 'bold'

                    # Add text to plot
                    ax.text(x, y, t, color=col, ha='center', va='center', fontdict=dict(size=value_font_size), **kw)

    # Add the aspect ===================================================================================================
    if aspect == 'auto':
        # TODO do something with the maximum number of characters in the text?
        aspect = 0.7
    ax.set_aspect(aspect=aspect)

    # Add NA mask ======================================================================================================
    if na_mask.sum().sum() >= 1:
        # there are NA values
        make_heatmap(data=na_mask.astype(float), show_labels=False, show_values=False,
                     cmap=ListedColormap([(0, 0, 0, 0), na_color]), ax=ax, aspect=aspect)

    # Add the labels ===================================================================================================
    if isinstance(show_labels, bool):
        show_x_labels = show_y_labels = show_labels
    elif isinstance(show_labels, str):
        if show_labels == 'x':
            show_x_labels = True
            show_y_labels = False
        elif show_labels == 'y':
            show_x_labels = False
            show_y_labels = True
        else:
            raise NotImplementedError(f'Unknown str value for show_labels {show_labels}')
    else:
        raise NotImplementedError(f'Unknown type for show_labels {type(show_labels)}')

    if show_x_labels:
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(data.columns, fontdict=dict(size=label_font_size), rotation=x_rotation)
    else:
        ax.set_xticks([])

    if show_y_labels:
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(data.index, fontdict=dict(size=label_font_size))
    else:
        ax.set_yticks([])

    # Return ===========================================================================================================
    if f is None:
        return ax
    else:
        return f, ax


def z_value(ci_level):
    """
    Compute the z-value that belongs to the confidence interval level

    Parameters
    ----------
    ci_level: float
        Confidence interval level, between 0 and 1

    Returns
    -------
    z: float
        Z-value of the confidence interval
    """
    return stats.norm.interval(ci_level, 0, 1)[1]


def ci(collection, ci_level, skip_na=False):
    std = np.std(collection)

    if skip_na:
        return std_n_to_ci(std, (~collection.isna()).sum(), ci_level)
    else:
        return std_n_to_ci(std, len(collection), ci_level)


def std_n_to_ci(std, n, ci_level):
    """
    Compute the confidence interval value(s), given the standard deviation(s), population size(s), and confidence
    interval level.

    Parameters
    ----------
    std: np.array or float or pd.Series of float
        Standard deviation(s)
    n: int or pd.Series of int
        Population size(s)
    ci_level: float
        Desired confidence interval level

    Returns
    -------
    ci_values: float or pd.Series of float
        ci_values (what comes after the +-)

    Notes
    -----
    If 'n' or 'std' is a Series, 'ci_values' will be a series with the same index. If both 'std' and 'n'are series, they
    need to have the same index.
    """
    if isinstance(std, pd.DataFrame) and isinstance(n, pd.Series):
        assert (std.index == n.index).all()
        return std.multiply(z_value(ci_level)).divide(n.pow(0.5), axis=0)

    return z_value(ci_level) * std / (n ** 0.5)


def get_mean_and_ci_gb(collection, ci_level, skip_na=False):
    res = get_mean_and_ci(collection, ci_level, skip_na)

    data = {
        'mean': res[0],
        f'ci{ci_level}': res[1],
    }

    if isinstance(collection, pd.DataFrame):
        return pd.DataFrame(data=data).stack()
    elif isinstance(collection, pd.Series):
        return pd.Series(data=data)
    else:
        raise NotImplementedError()


def mean_and_ci_gb_function(ci_level, skip_na=False):
    def fun(collection):
        return get_mean_and_ci_gb(collection, ci_level, skip_na)

    return fun


def mean_and_ci_tex_gb_function(ci_level, **kwargs):
    def fun(collection):
        return latex_string(collection, ci_level, **kwargs)

    return fun


def ci_gb_function(ci_level, skip_na):
    def fun(collection):
        return ci(collection, ci_level, skip_na)

    return fun


def get_mean_and_ci(collection, ci_level, skip_na=False):
    """
    Compute the mean and confidence interval of a given collection for a given ci_level.

    Parameters
    ----------
    collection: Iterable of Number
        The data for which to compute the mean and CI
    ci_level: float
        The confidence level to compute. Should be in [0,1]
    skip_na: bool
        Whether to account for missing values in the stddev to compute the ci

    Returns
    -------
    mean: Number
        The mean of the collection
    ci: Number
        The confidence interval of the collection given the ci_level. i.e. We have mean +- ci as the ci_level confidence
        interval.
    """
    return np.mean(collection), ci(collection, ci_level, skip_na)


def latex_string_from_m_ci(m, s, formatter='{:.2f}'):
    if isinstance(m, pd.Series) and isinstance(s, pd.Series):
        res = pd.Series(dtype=str, index=m.index)
        for idx in m.index:
            res[idx] = latex_string_from_m_ci(m[idx], s[idx])
        return res
    else:
        if pd.isna(m) or pd.isna(s):
            return '$N.A.$'
        else:
            return rf'${formatter.format(m)}\pm{formatter.format(s)}$'


def latex_string(collection, ci_level, formatter='{:.2f}', as_percentage=False, show_pct=True,
                 na_handling='ignore'):
    """

    Parameters
    ----------
    collection: Any
        Collection for which to compute mean/ci
    ci_level: float
        confidence level in [0,1]
    formatter: str or in
        How to format the mean / std. If int, uses this many decimals
    as_percentage: bool
        If True, multiply by 100 %
    show_pct: bool
        If False, do not add %. Ignore if as_percentage is False

    Returns
    -------

    """
    m, s = get_mean_and_ci(collection, ci_level)

    if isinstance(formatter, int):
        formatter = f'{{:.{formatter}f}}'

    if as_percentage:
        m *= 100
        s *= 100

    if isinstance(collection, pd.DataFrame):
        foo = pd.Series(dtype=str)
        for c in m.index:
            foo[c] = latex_string_from_m_ci(m[c], s[c], formatter)[:-1]
    else:
        foo = latex_string_from_m_ci(m, s, formatter)[:-1]

    if as_percentage and show_pct:
        foo += r"\%"

    return foo + '$'


def mean_ci(collection, ci_level=0.95, na_handling='error'):
    """
    Parameters
    ----------
    collection: Collection
        Collection of values
    ci_level: float
        Confidence Interval level, value in [0,1]
    na_handling: str
        How na values are handles. If NA values are found and na_handling has value:
        - 'error': an error is raised
        - 'propagate': both mean and ci are returned as NA on encountering
        - 'ignore': Na values are filtered out

    Returns
    -------
    mean: Any
        The mean of the collection
    ci: Any
        The confidence interval of the collection for the given ci_level
    """
    if isinstance(collection, pd.DataFrame):
        sr_m = pd.Series(dtype=np.float64)
        sr_c = pd.Series(dtype=np.float64)
        for name, sr in collection.iteritems():
            sr_m[name], sr_c[name] = mean_ci(sr, ci_level, na_handling)
        return sr_m, sr_c

    na_mask = pd.isna(collection)
    num_na = na_mask.sum()
    if na_handling == 'propagate':
        if any(na_mask):
            return pd.NA, pd.NA
        else:
            pass
    elif na_handling == 'ignore':
        collection = collection[~na_mask]
    elif na_handling == 'error':
        if num_na == 0:
            pass
        else:
            raise ValueError(f'Collection contains {num_na} NA values')
    else:
        raise NotImplementedError(na_handling)

    mean = np.mean(collection)
    std = np.std(collection)
    return mean, std_n_to_ci(std, len(collection), ci_level)


def mean_ci_gb(**kwargs):
    def fun(x):
        return mean_ci(x, **kwargs)

    return fun


def mean_ci_tex(collection, formatter=2, as_percentage=False, show_pct=False, **kwargs):
    m, s = mean_ci(collection, **kwargs)

    if isinstance(formatter, int):
        formatter = f'{{:.{formatter}f}}'

    if as_percentage:
        m *= 100
        s *= 100

    if isinstance(collection, pd.DataFrame):
        foo = pd.Series(dtype=str)
        for c in m.index:
            foo[c] = latex_string_from_m_ci(m[c], s[c], formatter)[:-1]
    else:
        if pd.isna(m) or pd.isna(s):
            return '$N.A.$'
        else:
            foo = latex_string_from_m_ci(m, s, formatter)[:-1]

    if as_percentage and show_pct:
        foo += r"\%"

    return foo + '$'


def mean_ci_tex_gb(**kwargs):
    def fun(x):
        return mean_ci_tex(x, **kwargs)

    return fun


# This is a WIP separate package used by some other projects
try:
    from functions.heatmap import make_heatmap
    from functions.confidence_interval import *
    from functions.general_functions import set_compare
except ModuleNotFoundError:
    pass
