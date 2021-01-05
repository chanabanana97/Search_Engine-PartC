# you can change whatever you want in this module, just make sure it doesn't 
# break the searcher module
import numpy as np
from numpy.linalg import norm


class RankerGlove:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_docs(relevant_docs, docs_as_vectors, query_vector, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param docs_as_vectors:
        :param query_vector:
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        docs_to_return = []
        for tweet_id in relevant_docs:
            if tweet_id in docs_as_vectors:
                cos_sim = np.dot(docs_as_vectors[tweet_id], query_vector) / (
                            norm(docs_as_vectors[tweet_id]) * norm(query_vector))
                if cos_sim > 0.98:
                    docs_to_return.append((tweet_id, cos_sim))
            # else:
            #     docs_to_return.append((tweet_id, 0.9))


        ranked_results = sorted(docs_to_return, key=lambda element: element[1], reverse=True)
        if k is not None:
            ranked_results = ranked_results[:k]
        return [d[0] for d in ranked_results]

# cosSim ranker
class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_docs(relevant_docs, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.

        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """

        ranked_results = sorted(relevant_docs, key=lambda element: element[1], reverse=True)
        if k is not None:
            ranked_results = ranked_results[:k]
        return [d[0] for d in ranked_results]




