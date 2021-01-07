import gensim.models

class Word2Vec:
    def __init__(self):

        input_file = 'glove.txt'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(input_file, binary=False)

    def update(self, array_of_words):
        similar_words = set(array_of_words)
        for word in array_of_words:
            try:
                for similar in self.model.similar_by_word(word,topn=5):
                    similar_words.add(similar[0])
            except KeyError:
                pass
        return list(similar_words)