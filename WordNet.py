from nltk.corpus import wordnet


class wordNet:
    def synsetsWord(self, Array_of_words):
        synonyms = set()
        for word in Array_of_words:
            print(wordnet.synsets(word))
            for syncs in wordnet.synsets(word):
                sy = syncs.lemmas()[0].name()
                synonyms.add(sy)
        return [synonyms]
