# you can change whatever you want in this module, just make sure it doesn't 
# break the searcher module


# cosSim ranker
class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_docs(relevant_docs, data, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.

        :param data: tuple (inverted_idx, posting_dict, num_of_documents)
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """

        docs_dict = data[1]
        for tweet_id in relevant_docs:
            term_dict = docs_dict[tweet_id]
            for term in term_dict:
                pass









        ranked_results = sorted(relevant_docs, key=lambda element: element[1], reverse=True)
        if k is not None:
            ranked_results = ranked_results[:k]
        return [d[0] for d in ranked_results]




