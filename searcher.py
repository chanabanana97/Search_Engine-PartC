from ranker import Ranker
import utils

# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None, method=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self.method = method
        self.our_data = self._indexer.data # tuple (index_bench, docs ,num_of_documents)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        query_as_list = self._parser.parse_sentence(query)
        if self.method is not None:
            query_as_list = self.method.update(query_as_list) # query expansion depending on method

        relevant_docs = self.relevant_docs_from_posting(query_as_list)

        ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, self.our_data, query_as_list, k)

        return len(ranked_doc_ids), ranked_doc_ids

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def relevant_docs_from_posting(self, query_as_list):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """

        relevant_docs = {}
        for term in query_as_list:
            if term in self.our_data[0]:
                posting_list = self.our_data[0][term][1]
                for doc_id in posting_list:
                    df = relevant_docs.get(doc_id, 0)
                    relevant_docs[doc_id] = df + 1
        return relevant_docs
