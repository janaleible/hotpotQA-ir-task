"""This package implements retrieval methods for the HotpotQA dataset starting from the full raw wiki dataset.
Contains:
    -- packages for baseline and more advanced systems.
    -- an interface to the Indri index using pyndri in ``index.py`.
    -- a tokenizer to convert queries to a format acceptable by pyndri in ``tokenizer.py``
"""

from retrieval import baseline
