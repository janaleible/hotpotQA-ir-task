import concurrent.futures
import itertools
from typing import Callable, Iterable


def chunk(n, iterable):
    """Collect data into fixed-length chunks or blocks, enumerate them, and filter out fill values.
            grouper('ABCDEFG', 3, 'x') --> (0, ABC) (1, DEF) (2, G)"
    """
    args = [iter(iterable)] * n
    batches = itertools.zip_longest(*args, fillvalue=None)
    enumerated_and_filtered = []
    for idx, batch in enumerate(batches):
        enumerated_and_filtered.append((idx, tuple(filter(lambda elem: elem is not None, batch))))
    return enumerated_and_filtered


def execute(func: Callable, items: Iterable):
    """Execute a callable over the iterable in parallel."""

    with concurrent.futures.ProcessPoolExecutor() as pool:
        return pool.map(func, items, timeout=60 * 5)
