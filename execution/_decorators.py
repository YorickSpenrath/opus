from ..objects import AbstractOpus
from ..objects.multi_opus import AbstractMultiOpus


# This allows functions that operate on only an AbstractOpus to operate directly on all AbstractOpus instances of
# an OpusExperiment (essentially the equivalent of vectorize)

def run_all(func):
    def wrapper_run(source: [AbstractMultiOpus, AbstractOpus]):
        if isinstance(source, AbstractMultiOpus):
            for opus in source:
                func(opus)
        else:
            func(source)

    return wrapper_run


def verify_all(msg):
    def inner(func):
        def wrapper_verify(source):
            print(f'Verifying {msg}')
            if isinstance(source, AbstractMultiOpus):
                for opus in source:
                    if not func(opus):
                        return False
            return True

        return wrapper_verify

    return inner
