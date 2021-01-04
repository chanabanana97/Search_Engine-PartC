from autocorrect import Speller

class spell_checker:
    def __init__(self):
        self.spell = Speller(lang='en')

    def correct_spelling(self, list_to_correct:list):
        corrected_list = []
        for word in list_to_correct:
            corrected_list.append(self.spell(word))
        return corrected_list