"""This package implements retrieval methods for the HotpotQA dataset starting from the full raw wiki dataset.
Contains:
    -- packages for baseline and more advanced systems.
    -- an interface to the Indri index using pyndri in ``index.py`.
"""

from retrieval import baseline
from retrieval import filters
from retrieval import evaluate