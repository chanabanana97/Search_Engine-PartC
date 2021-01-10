import numpy as np


class Glove:
    def __init__(self, glove_file):
        self.vector_dict = {}
        self.vectors_for_doc = [np.zeros(100)]
        self.doc_value_dict = {}  # key: doc(tweet id), value: avg vector - value of doc

        lines = glove_file.readlines()[1:]
        for line in lines:
            records = line.split()
            word = records[0]
            vector = np.asarray(records[1:], dtype='float32')
            self.vector_dict[word] = vector
        glove_file.close()

    def doc_to_vec(self, tweet_term_dict):

        for term in tweet_term_dict:
            if term in self.vector_dict:
                self.vectors_for_doc.append(self.vector_dict[term])
        # if np.add.reduce(vectors_for_doc) is not np.nan or len(vectors_for_doc) != 0:
        return np.add.reduce(self.vectors_for_doc) / len(self.vectors_for_doc)



