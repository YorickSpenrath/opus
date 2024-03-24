import itertools

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..functions import set_compare
from ._decorators import run_all, verify_all
from ..objects import AbstractOpus
from ..objects.multi_opus import AbstractMultiOpus
from ..objects.single_opus import SingleOpus
from ..strings import idx

idx_enc = [f'{c}_encoding' for c in idx]


def get_tpm_dataframe(ao: AbstractOpus):
    res = pd.DataFrame()
    print(f'Computing TPM table {ao}')
    for e in ao.experiment_iterator():
        e: SingleOpus

        experiment_features = ao.load_features(e)
        experiment_features.name = e.index

        # Load encodings
        experiment_hps_encodings = pd.Series(data=[e.ts, e.data_reference.encoding, e.model_name, e.strategy],
                                             index=idx_enc,
                                             name=e.index)

        res = pd.concat(
            [res, pd.concat([experiment_features.to_frame().T, experiment_hps_encodings.to_frame().T], axis=1)], axis=0)

    res.index.names = idx
    return res


@run_all
def recompute_tpm_df(opus: AbstractOpus):
    df = get_tpm_dataframe(opus)
    opus.store_tpm_df(df)


def precompute_tpm_df(opus: AbstractOpus):
    if not opus.has_tpm_df():
        df = get_tpm_dataframe(opus)
        opus.store_tpm_df(df)


def split_and_preprocess_tpm(ao: AbstractOpus, metric: str, ohe=None):
    # Get or compute training data -------------------------------------------------------------------------------------
    if not ao.has_tpm_df():
        res = get_tpm_dataframe(ao)
        ao.store_tpm_df(res)
    else:
        res = ao.load_tpm_df()

    # Categorical x values ---------------------------------------------------------------------------------------------
    if ohe is None:
        if not ao.has_tpm_ohe():
            # Not given, not stored: Compute and store
            ohe = OneHotEncoder()
            ohe.fit(res[idx_enc])
            ao.store_tpm_ohe(ohe)
        else:
            ohe = ao.load_tpm_ohe()
    else:
        pass

    data = ohe.transform(res[idx_enc].astype('O'))
    x = pd.DataFrame(index=res.index, data=data.todense(), columns=ohe.get_feature_names_out(idx_enc)).astype(int)
    res = res.drop(columns=idx_enc)

    # Numerical values -----------------------------------------------------------------------------------------------
    y = res[f'oth_{metric}_{ao.tt_string(False)}']

    # Use oth and metric for phase 1, and a(n)cp for phase 1 and 2
    # TODO: in the old version, all metrics were used. The difficulty with this is that the models are invalidated
    #  once you add a new metric, which means you have to recompute all TPM thresholds and scores
    x_columns = [
        f'oth_{metric}_{ao.tt_string(1)}',
        f'{metric}_{ao.tt_string(1)}',
        f'ancp_{ao.tt_string(1)}',
        f'ancp_{ao.tt_string(2)}',
        f'acp_{ao.tt_string(1)}',
        f'acp_{ao.tt_string(2)}'
    ]

    x = pd.concat([x, res[x_columns]], axis=1)

    return x, y


@run_all
def run(ao: [AbstractOpus, AbstractMultiOpus], redo=False):
    for model_name, model_constructor in ao.tpm_model_dict.items():
        for metric in ao.metric_methods_dict.keys():
            if not redo and ao.has_tpm(model_name=model_name, metric=metric):
                continue

            print(f'[{ao}] Computing Threshold Prediction Model {model_name} for metric {metric}')
            x, y = split_and_preprocess_tpm(ao, metric)

            model = model_constructor()
            mask = ~x.isna().any(axis=1)
            model.fit(x[mask], y[mask])
            ao.store_tpm(model, model_name, metric=metric)


@verify_all('02-TPMs')
def verify(ao: [AbstractOpus, AbstractMultiOpus]):
    print(f'Verifying 02-TPMS of {ao}: ', end='', flush=False)
    for (base_model, metric) in itertools.product(ao.tpm_model_dict.keys(), ao.metric_methods_dict):
        if not ao.has_tpm(base_model, metric):
            print(f'missing TPM {base_model}-{metric}')
            return False

    if not ao.has_tpm_ohe():
        print('missing OHE')
        return False

    df = ao.load_tpm_df()

    try:
        df.loc[ao.experiment_dataframe_index]
    except KeyError:
        print('not all experiments in TPM df')
        return False

    expected_columns = []
    for phase in range(1, 4):
        expected_columns.append(f'ancp_phase{phase}')
        expected_columns.append(f'acp_phase{phase}')
        for metric in ao.metric_methods_dict.keys():
            expected_columns.append(f'oth_{metric}_phase{phase}')
            expected_columns.append(f'{metric}_phase{phase}')

    expected_columns += idx_enc

    missing, superfluous, okay = set_compare(df.columns, expected_columns)

    if missing:
        print(f'tpm columns: missing {missing}')
        return False

    print('All good')
    return True
