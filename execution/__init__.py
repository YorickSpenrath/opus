from ..objects.multi_opus import AbstractMultiOpus


def full_run(coe: AbstractMultiOpus):
    from .compute1_A_probabilities import run
    run(coe)

    from .compute1_B_features import run
    run(coe)

    from .compute2_threshold_prediction_models import run
    run(coe)

    from .compute3_adapted_thresholds import run
    run(coe)

    from .compute4a_scores import run
    run(coe)

    from .compute4b_alternative_method_scores import run
    run(coe)

    from .compute4c_competitor_scores import run
    run(coe)


def full_verify(coe: AbstractMultiOpus):
    from .compute1_A_probabilities import verify
    verify(coe)

    from .compute1_B_features import verify
    verify(coe)

    from .compute2_threshold_prediction_models import verify
    verify(coe)

    from .compute3_adapted_thresholds import verify
    verify(coe)

    from .compute4a_scores import verify
    verify(coe)

    from .compute4b_alternative_method_scores import verify
    verify(coe)

    from .compute4c_competitor_scores import verify
    verify(coe)
