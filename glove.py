import numpy as np
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec


class Glove:
    def __init__(self):
        self.vector_dict = {}
        # self.vectors_for_doc = [np.zeros(25)]
        self.vectors_for_doc = [np.zeros(100)]
        self.doc_value_dict = {}  # key: doc(tweet id), value: avg vector - value of doc
        glove_file = open('C:\\Users\\elitm\\PycharmProjects\\Search_Engine-PartC\\glove.twitter.27B.25d.txt', encoding="utf8")
        # glove_file = open('glove.txt', encoding="utf8")
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector = np.asarray(records[1:], dtype='float32')
            self.vector_dict[word] = vector
        glove_file.close()
        # glove_input_file = '../../../../glove.twitter.27B.25d.txt'
        # glove_input_file = 'glove.txt'
        # # word2vec_output_file = 'glove.twitter.27B.25d.txt.word2vec'
        # word2vec_output_file = 'glove.txt.word2vec'
        # glove2word2vec(glove_input_file, word2vec_output_file)
        # self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    def doc_to_vec(self, tweet_term_dict):

        for term in tweet_term_dict:
            if term in self.vector_dict:
                self.vectors_for_doc.append(self.vector_dict[term])
        # if np.add.reduce(vectors_for_doc) is not np.nan or len(vectors_for_doc) != 0:
        return np.add.reduce(self.vectors_for_doc) / len(self.vectors_for_doc)



