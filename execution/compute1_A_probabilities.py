import pandas as pd

from ._decorators import verify_all, run_all
from .. import strings as ps
from ..objects import AbstractOpus, DataObject
from ..objects.multi_opus import AbstractMultiOpus


def get_positive_probabilities(probabilities, model):
    if hasattr(model, 'classes_'):
        if model.classes_[0]:
            probabilities = probabilities[:, 0]
        else:
            probabilities = 1 - probabilities[:, 0]
    else:
        probabilities = probabilities[:, 1]

    return probabilities


@run_all
def run(amo: [AbstractOpus, AbstractMultiOpus], redo=False):
    for e in amo.experiment_iterator():
        # Skip if done -------------------------------------------------------------------------------------------------
        if not redo and all(map(lambda phase: amo.has_probabilities(e, phase), [1, 2, 3])):
            continue
        print(f'OPUS {e.strategy} -> {e.ts} -> {e.data_reference} -> {e.model_name}')

        def compute_probabilities(xy_train: DataObject, xy_test: DataObject, phase: int):
            if amo.has_probabilities(e, phase):
                return
            else:
                # Correct for adding points from test/train
                if e.strategy == ps.UPDATE:
                    xy_train = xy_train + xy_test.converted_only

                # Train model and/or make predictions ------------------------------------------------------------------
                if xy_train.labels.all():
                    probabilities = pd.Series(index=xy_test.x.index, data=1)
                elif not xy_train.labels.any():
                    probabilities = pd.Series(index=xy_test.x.index, data=0)
                else:

                    model = amo.model_constructor_dict[e.model_name]()
                    try:
                        model.fit(xy_train.x.values, xy_train.labels.values)
                        amo.store_model(model, e, phase)
                        probabilities = model.predict_proba(xy_test.x.values)
                        probabilities = get_positive_probabilities(probabilities, model)
                    except TypeError as te:
                        if str(te) == "_new_learning_node() got an unexpected keyword argument 'is_active_node'" and \
                                e.model_name == ps.AHOT:
                            probabilities = pd.Series(index=xy_test.x.index, data=pd.NA)
                        else:
                            raise te

                probabilities = pd.Series(index=xy_test.x.index, data=probabilities, name=ps.PROBABILITIES)
                amo.store_probabilities(probabilities, e, phase)

        xy_phase1: DataObject = e.data_reference.phase_1_data
        xy_phase2: DataObject = e.data_reference.phase_2_data
        xy_phase2 = xy_phase2.match_features(xy_phase1)
        # Normalize if needed ------------------------------------------------------------------------------------------
        if amo.model_needs_normalized_data(e.model_name):
            xy_phase1, normalizer = xy_phase1.normalize()
            xy_phase2, _ = xy_phase2.normalize(normalizer)

        compute_probabilities(*xy_phase1.train_test_split(), phase=True)
        compute_probabilities(xy_phase1, xy_phase2, phase=False)
        compute_probabilities(*xy_phase2.train_test_split(), phase=3)


@verify_all('Probabilities')
def verify(opus_experiment: [AbstractOpus, AbstractMultiOpus]):
    # TODO: this only checks that probabilities are computed at all, not whether the correct ones are computed
    print(str(opus_experiment))
    for e in opus_experiment.experiment_iterator():
        if not opus_experiment.has_probabilities(e, True):
            return False
        if not opus_experiment.has_probabilities(e, False):
            return False
        if not opus_experiment.has_probabilities(e, 3):
            return False
    return True
