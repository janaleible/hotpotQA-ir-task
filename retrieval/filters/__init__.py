"""Baseline implementation of retrieval system according to https://arxiv.org/abs/1809.09600
``filter`` is the top 5000 unigram bigram strategy used by the authors.
``ranking`` is the top 10 documents ranked according to bigram tf-idf.
"""
from retrieval.filters import overlap
from retrieval.filters import uni_tfidf
from retrieval.filters import bi_tfidf
from retrieval.filters import prf_lm

__all__ = [
    'overlap',
    'uni_tfidf',
    'bi_tfidf',
    'prf_lm'
]
