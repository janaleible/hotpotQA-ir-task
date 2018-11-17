import concurrent.futures
import itertools
from typing import Callable, Iterable


def chunk(n, iterable):
    it = iter(iterable)
    if n > len(iterable):
        yield tuple(it)

        return

    while True:
        ch = tuple(itertools.islice(it, n))
        if not ch:
            return

        yield ch


def execute(func: Callable, items: Iterable):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        return executor.map(func, items)
