import string
from collections import Counter
import nltk
from unidecode import unidecode

from retrieval.index import Index


def unigram_bigram_filter(query: str, index: Index, n: int = 5000):
    """ Retrieves the at most n candidates from the full set of articles based on query-document pair bigram/unigram
    matches. Uses pre-built inverted index. Assumed to be equivalent to Algorithm 2, Appendix C of HotpotQA paper.
    Possible mismatches:
        -- unigram/bigrams counts in our case are considered only over first at most 500 characters of the article. Not
        clear if they use full article or not.
        -- implementation does not follow algorithm exactly since that seems very inefficient. We made it better, but
        maybe some edge-cases result in different results.

    :param query: A string of words to match.
    :param index: The prebuilt inverted index.
    :param n: The control threshold
    :return: A list of at most 5000 candidates.
    """

    # tokenize, step, filter stopwords and collect unigrams and bigrams

    query = unidecode(query)
    query = query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    query = query.lower()

    tokenized_query = [token for token in nltk.word_tokenize(query) if token not in index.stopwords]
    query_unigrams = set([(token,) for token in tokenized_query])
    query_bigrams = set(nltk.bigrams(tokenized_query))

    # count the overlapping n-gram for each query-document pair
    overlap_set = Counter()
    for bigram in query_bigrams:
        for (doc_id, _) in index.bigram_lookup(bigram[0], bigram[1]):
            overlap_set[doc_id] += 1

    for unigram in query_unigrams:
        for (doc_id, _) in index.unigram_lookup(str(unigram[0])):
            overlap_set[doc_id] += 1

    # Get the best n+1 documents and filter all the ones that have a count equal to the smallest count in the list.
    most_common = overlap_set.most_common(n + 1)
    candidates = filter(lambda t: t[1] > most_common[-1][1], most_common)

    return [doc_id for (doc_id, count) in candidates]
