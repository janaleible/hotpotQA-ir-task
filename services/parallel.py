import concurrent.futures
import itertools
from typing import Callable, Iterable


def chunk(n, iterable):
    """Collect data into fixed-length chunks or blocks
            grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=None)


def execute(func: Callable, items: Iterable):
    """Execute a callable over the iterable in parallel."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        return executor.map(func, items)
