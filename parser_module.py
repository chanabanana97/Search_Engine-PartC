from nltk.tokenize import word_tokenize
import re
from document import Document

class Parse:

    def __init__(self):

        with open('stop_words.txt', 'r') as f:
            self.our_stop_words = f.read().splitlines()
        self.stop_words_dict = {key: None for key in self.our_stop_words}

    def handle_hashtag(self, hashtag_str: str):
        glue = ' '
        if hashtag_str.__contains__("_"):
            result = hashtag_str.split("_")
        else:  # separate by uppercase letters
            result = ''.join(glue + x.lower() if x.isupper() else x for x in hashtag_str).strip(glue).split(glue)
        result.append("#" + hashtag_str.lower().replace("_", ""))
        result = [i for i in result if len(i) > 1] # remove single letters
        return result

    def handle_numbers(self, num_as_str):
        num = int(float(num_as_str.replace(",", "")))
        if num < 1000:
            return num_as_str
        k = pow(10, 3)
        m = pow(10, 6)
        b = pow(10, 9)
        if k <= num < m:
            return str(int(num / k)) + "K"
        if m <= num < b:
            return str(int(num / m)) + "M"
        if num >= b:
            return str(int(num / b)) + "B"

    # our rule 1: remove emojis from tweets
    def remove_emojis(self, txt):
        re_emoji = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              "]+", flags=re.UNICODE)
        word = re_emoji.sub(r'', txt)
        return word

    def parse_sentence(self, text):
        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param text:
        :return:
        """

        if text is None:
            return []
        text_tokens = word_tokenize(text)
        text_tokens_without_stopwords = []
        # text_lower_tokens_without_stopwords = [w.lower() for w in text_tokens if w not in self.stop_words]

        # remove stopwords
        for w in text_tokens:
            if w.lower() not in self.stop_words_dict:
                text_tokens_without_stopwords.append(w)

        # parsing
        doc_length = len(text_tokens_without_stopwords)
        num_dict = {"thousand": "K", "million": "M", "billion": "B", "dollar": "$", "dollars": "$", "bucks":"$", "percent": "%",
                    "$": "$", "%": "%",
                    "percentage": "%"}

        similar_words_dict = {"corona":"covid", "covid19":"covid", "coronavirus":"covid", "covid-19":"covid", "covid": "covid","#covid": "covid", "#covid19": "covid"}

        new_tokenized_text = []
        i = -1
        # for i in range(doc_length):
        while i < doc_length - 1:
            # please note: when we do i += 1 it is because next_term(old_token[i + 1]) is used already so we skip over it next iteration
            # so we dont go over it twice

            i += 1
            term = text_tokens_without_stopwords[i]

            term = term.encode("ascii", "ignore").decode()  # remove ascii
            if term.lower() in similar_words_dict:
                new_tokenized_text.append(similar_words_dict[term.lower()])
                continue
            next_term = None
            if term.startswith("//t") or (term.isalpha() and len(term) == 1): # remove short urls and terms that are single letters
                continue
            if term.__contains__("-"):
                new_tokenized_text.extend(term.split("-"))
            if i + 1 < doc_length:
                next_term = text_tokens_without_stopwords[i + 1]
            if term is "@" and next_term is not None:
                i += 2 # removing @ and name
                continue
            if term is "#" and next_term is not None:
                new_tokenized_text.extend(self.handle_hashtag(next_term))
                i += 1
            elif term is "$" and next_term is not None and str.isdigit(
                    next_term.replace(",", "")):  # $100 thousand / $75 --> 100K$ / 75$
                num = self.handle_numbers(next_term)
                if i + 2 < doc_length and text_tokens_without_stopwords[i + 2] in num_dict:
                    num = num + num_dict[text_tokens_without_stopwords[i + 2]]
                    i += 1
                new_tokenized_text.append(num + "$")
                i += 1
            elif str.isdigit(term.replace(",", "")):  # if term is a number
                # deal with decimal number like 10.1234567 -> 10.123
                num = self.handle_numbers(term)
                if next_term is not None and next_term.lower() in num_dict:
                    new_tokenized_text.append(num + num_dict[next_term.lower()])
                    i += 1
                else:
                    new_tokenized_text.append(num)
            elif not term.isidentifier():  # identifier: (a-z) and (0-9), or underscores (_)
                emojis_removed = self.remove_emojis(term)
                if emojis_removed is not "":
                    new_tokenized_text.append(emojis_removed)
            else:
                new_tokenized_text.append(term.lower())

        return new_tokenized_text

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-preseting the tweet.
        :return: Document object with corresponding fields.
        """

        tweet_id = doc_as_list[0]
        tweet_date = doc_as_list[1]
        full_text = doc_as_list[2]
        url = doc_as_list[3]
        indice = doc_as_list[4]
        retweet_text = doc_as_list[5]
        retweet_url = doc_as_list[6]
        retweet_indice = doc_as_list[7]
        quote_text = doc_as_list[8]
        quote_url = doc_as_list[9]
        quoted_indice = doc_as_list[10]
        retweet_quoted_text = doc_as_list[11]
        retweet_quoted_url = doc_as_list[12]
        retweet_quoted_indice = doc_as_list[13]

        term_dict = {}

        tokenized_text = self.parse_sentence(full_text)
        tokenized_quote = self.parse_sentence(quote_text)
        # tokenized_url = self.handle_url(url)


        doc_length = len(tokenized_text)  # after text operations - length of full_text

        new_tokenized_text = tokenized_text + tokenized_quote

        # spell checker
        # new_tokenized_text = self.spell.update(new_tokenized_text)

        for term in new_tokenized_text:
            if term is not "":  # or (term.isalpha() and len(term) == 1)
                if term not in term_dict:
                    term_dict[term] = 1
                else:
                    term_dict[term] += 1

        document = Document(tweet_id, tweet_date, full_text, url, retweet_text, retweet_url, quote_text,
                            quote_url, term_dict, doc_length)
        return document
