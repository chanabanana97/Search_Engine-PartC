# you can change whatever you want in this module, just make sure it doesn't 
# break the searcher module


# cosSim ranker
import math


class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_docs(relevant_docs, data, query,k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.

        :param query: query: list of terms
        :param data: tuple (index_bench, docs ,num_of_documents)
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """

        docs_dict = data[1]
        num_of_docs = data[2]
        query_len = len(query)
        most_relevant_docs = {}
        for tweet_id in relevant_docs:
            tf_idf = 0
            mechane_cos_sim = 0
            term_dict = docs_dict[tweet_id]
            doc_len = len(term_dict) # length: amount of unique words in doc
            doc_max = max(term_dict.values())
            try:
                for term in term_dict:
                    inverted_idx = data[0][term][0]
                    if term in query:
                        if inverted_idx<100:
                            term_in_query = 0.52
                        else:
                            term_in_query = 0.48
                    else:
                        term_in_query = 0
                    posting_dict = data[0][term][1]
                    count_term_in_doc = posting_dict[tweet_id]
                    count_docs_with_term = inverted_idx
                    tf_idf += ((count_term_in_doc/(doc_len*0.05 + doc_max*0.95)) * math.log((num_of_docs/count_docs_with_term), 2) * term_in_query)
                    mechane_cos_sim += math.pow((count_term_in_doc/doc_len) * math.log((num_of_docs/count_docs_with_term), 2), 2)
                if math.sqrt(mechane_cos_sim * query_len) == 0:
                    continue
                cos_sim = tf_idf / math.sqrt(mechane_cos_sim * query_len)
                if cos_sim > 0.145:
                    most_relevant_docs[tweet_id] = cos_sim
                # most_relevant_docs[tweet_id] = tf_idf
            except KeyError:
                pass


        ranked_results = sorted(most_relevant_docs.items(), key=lambda element: element[1], reverse=True)
        ranked_results = ranked_results[:k]
        return [d[0] for d in ranked_results]




