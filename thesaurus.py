from nltk.corpus import lin_thesaurus as thes

class Thesaurus:

    def __init__(self):
        pass

    def update(self, array_of_words):
        synonyms = set(array_of_words)
        for word in array_of_words:
            syn_list = list(thes.synonyms(word, fileid="simN.lsp"))
            for syn in syn_list[:2]:
                synonyms.add(syn)
        return list(synonyms)