from nltk.corpus import wordnet


class WordNet:
    def update(self, array_of_words):
        synonyms = set(array_of_words)
        for word in array_of_words:
            for syncs in wordnet.synsets(word):
                sy = syncs.lemmas()[0].name()
                synonyms.add(sy)
        return list(synonyms)
