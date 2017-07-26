import csv, logging, re, nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import feature_extraction
import tqdm


class FeatureData(object):
    def __init__(self, article_file_path, stances_file_path):
        self.number_of_classes = 4
        self.classes = ['agree', 'disagree', 'discuss', 'unrelated']
        self.articles = self._get_articles(article_file_path)  # list of dictionaries
        self.stances = self._get_stances(stances_file_path)
        self.number_of_stances = len(self.stances)
        self.number_of_articles = len(self.articles)

    def get_clean_articles(self):
        """Returns a dictionary with Body ID's as keys and article bodies as values."""
        clean_articles = []
        logging.debug('Retrieving clean articles...')

        for item in tqdm.tqdm(self.articles):
            cleaned_article = clean(item['articleBody'])
            tokens = tokenize_text(cleaned_article)
            no_stop_word_tokens = remove_stopwords(tokens)
            lemmatized_tokens = get_tokenized_lemmas(no_stop_word_tokens)
            clean_articles.append({'articleBody': ' '.join(lemmatized_tokens),
                                   'Body ID': item['Body ID']})
        return {article['Body ID']: article['articleBody'] for article in clean_articles}

    #We need the stop words for POS tagging to work propperly
    def get_original_articles(self):
        clean_articles = []
        logging.debug('Retrieving original articles...')
        for item in tqdm.tqdm(self.articles):
            #cleaned_article = clean(item['articleBody'])
            cleaned_article = item['articleBody'].decode('unicode_escape').encode('ascii', 'ignore')
            clean_articles.append({'articleBody':cleaned_article,
                                   'Body ID': item['Body ID']})
        return {article['Body ID']: article['articleBody'] for article in clean_articles}

    def get_clean_stances(self):
        """Retrieves a list of dictionaries containing the fully cleaned Headlines and the Body ID and Stance for
        each headline."""
        clean_headlines = []
        logging.debug('Retrieving clean stances...')

        for item in tqdm.tqdm(self.stances):
            cleaned_headline = clean(item['Headline'])
            tokens = tokenize_text(cleaned_headline)
            no_stop_word_tokens = remove_stopwords(tokens)
            lemmatized_tokens = get_tokenized_lemmas(no_stop_word_tokens)
            clean_headlines.append({'Headline': ' '.join(lemmatized_tokens),
                                    'originalHeadline': cleaned_headline,
                                    'Body ID': item['Body ID'],
                                    'Stance': item['Stance']})

        return clean_headlines

    def _get_articles(self, path):
        # Body ID, articleBody
        articles = []
        with open(path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                row['Body ID'] = int(row['Body ID'])
                articles.append(row)
        return articles

    def _get_stances(self, path):
        # Headline, Body ID, Stance
        stances = []
        with open(path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                row['Body ID'] = int(row['Body ID'])
                stances.append(row)
        return stances


def normalize_word(w):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(w).lower()


def clean(text):
    return " ".join(re.findall(r'\w+', text.decode('utf-8'), flags=re.UNICODE)).lower()


def tokenize_text(text):
    return [token for token in word_tokenize(text)]


def remove_stopwords(list_of_tokens):
    return [word for word in list_of_tokens if word not in feature_extraction.text.ENGLISH_STOP_WORDS]


def get_tokenized_lemmas(tokens):
    return [normalize_word(token) for token in tokens]


if __name__ == '__main__':
    fd = FeatureData('data/train_bodies.csv', 'data/train_stances.csv')
    cleaned_articles = fd.get_clean_articles()
    original_articles = fd.get_original_articles()
    cleaned_stances = fd.get_clean_stances()

    print cleaned_articles[0]
    print cleaned_articles[4]
    print cleaned_stances[0]
