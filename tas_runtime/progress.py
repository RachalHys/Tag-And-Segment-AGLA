"""Named research stages and text progress bars forwarded by the notebook."""

from contextlib import contextmanager
import sys
import time

from tqdm import tqdm


def progress(iterable=None, *, desc, **kwargs):
    return tqdm(
        iterable, desc=desc, file=sys.stdout, ascii=True, ncols=100,
        mininterval=1, position=0, **kwargs,
    )


@contextmanager
def step(label):
    started = time.perf_counter()
    print(f"[stage] {label}", flush=True)
    try:
        yield
    except BaseException:
        print(f"[stage] {label}: stopped after {time.perf_counter() - started:.1f}s", flush=True)
        raise
    else:
        print(f"[stage] {label}: done in {time.perf_counter() - started:.1f}s", flush=True)
