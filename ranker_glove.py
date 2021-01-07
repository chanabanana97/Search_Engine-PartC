import numpy as np
from numpy.linalg import norm
from glove import Glove


class RankerGlove:
    def __init__(self):
        self.glove = Glove()


    # @staticmethod
    def rank_relevant_docs(self, relevant_docs, data, query, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """

        docs_dict = data[1]
        docs_as_vectors = {}

        query_vector = self.glove.doc_to_vec(query)
        for tweet_id in relevant_docs:
            docs_as_vectors[tweet_id] = self.glove.doc_to_vec(docs_dict[tweet_id])

        docs_to_return = []
        for tweet_id in relevant_docs:
            if query_vector is not np.nan and docs_as_vectors[tweet_id] is not np.nan:
                cos_sim = np.dot(docs_as_vectors[tweet_id], query_vector) / (
                        norm(docs_as_vectors[tweet_id]) * norm(query_vector))
                # if cos_sim > 0.5:
                docs_to_return.append((tweet_id, cos_sim))

        ranked_results = sorted(docs_to_return, key=lambda element: element[1], reverse=True)

        # k = min(k, len(ranked_results))
        ranked_results = ranked_results[:k]

        return [d[0] for d in ranked_results]
