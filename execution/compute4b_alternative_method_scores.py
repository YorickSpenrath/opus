from ._decorators import run_all
from .. import strings as ps
from ..objects import AbstractOpus
from ..objects.multi_opus import AbstractMultiOpus
from ..functions import set_compare


@run_all
def run(ao: [AbstractOpus, AbstractMultiOpus], redo=False):
    print(f'Computing alternative scores for {ao}')
    if not redo and ao.alternative_method_scores_completed:
        print('\tAll scores were computed already')
        return
    res = ao.compute_alternative_method_scores()
    for metric, scores in res.items():
        ao.store_score_type(metric=metric, score_type=ps.ALTERNATIVE, scores=scores)


def verify(oe: AbstractMultiOpus):
    # TODO: you don't know which ones to expect, this depends on the opus instances
    for ao in oe:
        print(f'Verifying 04B-Alternative Scores for {ao}: ', end='', flush=True)
        for metric in [ps.NAPE_CONVERTED_OBJECT_COUNT]:
            if not ao.has_score_type(metric=metric, score_type=ps.ALTERNATIVE):
                print(f'missing alternative scores {metric}')
                return False
            df = ao.load_score_type(metric=metric, score_type=ps.ALTERNATIVE)
            missing, superfluous, okay = set_compare(df.index, ao.all_timesteps)
            if not okay:
                print(f'timesteps error {metric}, missing = {missing}, superfluous={superfluous}')

        print('All good')
    return True
