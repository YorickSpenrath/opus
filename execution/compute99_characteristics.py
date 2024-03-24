import matplotlib.pyplot as plt

import themes
from ..objects import AbstractOpus
from .. import strings as ps


def make_encodings_plot(ao: AbstractOpus):
    df = ao.load_tpm_df()

    res = df.reset_index()[[ps.TIMESTEP, ps.FEATURE_INDEX]].drop_duplicates().groupby(ps.TIMESTEP).size()

    f, ax = plt.subplots()
    ax.bar(range(len(res)), res.values, color=themes.BRIGHT_ORANGE)

    ax.set_ylabel('Number of encodings')
    ax.set_yticks([])
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels(res.index)
    ax.set_xlabel(ao.timestep_name)

    for i in range(len(res)):
        ax.text(x=i,
                y=res.iloc[i],
                va='bottom',
                ha='center',
                s=str(res.iloc[i]))
    print(res.sum())

    ao.store_characteristic_figure(f, 'encoding_count')
    plt.show()
