import utils
from glove import Glove

# DO NOT MODIFY CLASS NAME
class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config, is_glove=False):
        self.inverted_idx = {}
        self.postingDict = {}
        self.idx_bench = {}
        self.config = config
        self.is_glove = is_glove
        # for glove
        if self.is_glove:
            self.docs_as_vectors = {}
            self.glove = Glove()

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """

        document_dictionary = document.term_doc_dictionary
        # Go over each term in the doc
        for term in document_dictionary:
            try:
                # Update inverted index and posting
                if term not in self.inverted_idx:
                    self.inverted_idx[term] = 1
                else:
                    self.inverted_idx[term] += 1

                if not self._is_term_exist(term):
                    self.postingDict[term] = []

                self.postingDict[term].append((document.tweet_id, document_dictionary[term]))

                self.idx_bench[term] = (self.inverted_idx[term], self.postingDict[term])

            except:
                print('problem with the following key {}'.format(term[0]))
        if self.is_glove:
            self.docs_as_vectors[document.tweet_id] = self.glove.doc_to_vec(document_dictionary)

    def remove_uncommon_words(self):
        """
        removes words from idx_bench with quantity of 1 and updates idx_bench
        :return:
        """
        for term in list(self.idx_bench):
            if self.idx_bench[term][0] == 1:
                del self.idx_bench[term]
                del self.postingDict[term]
                del self.inverted_idx[term]

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        utils.load_obj(fn)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        utils.save_obj(fn, "idx_bench") # TODO change after?

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.postingDict

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist(term) else []
        # return self.idx_bench[term][1] if self._is_term_exist(term) else []
