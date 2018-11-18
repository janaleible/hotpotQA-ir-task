"""Baseline implementation of retrieval system according to https://arxiv.org/abs/1809.09600
``filter`` is the top 5000 unigram bigram strategy used by the authors.
``ranking`` is the top 10 documents ranked according to bigram tf-idf.
"""
from retrieval.baseline import filter
from retrieval.baseline import ranking

__all__ = [
    'filter',
    'ranking'
]
