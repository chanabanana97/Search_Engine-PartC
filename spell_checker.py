from spellchecker import SpellChecker

class Spell_Checker:
    def __init__(self):
        # self.spell = Speller(lang='en')
        self.spell = SpellChecker(language='en')
        self.spell.word_frequency.add('covid')
        self.spell.word_frequency.add('covid19')
        self.spell.distance = 1

    def update(self, list_to_correct:list):
        corrected_list = []
        for word in list_to_correct:
            corrected_list.append(self.spell.correction(word))
        return corrected_list

