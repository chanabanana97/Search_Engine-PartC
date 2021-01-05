from nltk.corpus import wordnet


class WordNet:
    def update(self, array_of_words):
        synonyms = set()
        for word in array_of_words:
            print(wordnet.synsets(word))
            for syncs in wordnet.synsets(word):
                sy = syncs.lemmas()[0].name()
                synonyms.add(sy)
        return list(synonyms)
