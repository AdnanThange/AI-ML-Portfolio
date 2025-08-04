import re
import string
from typing import Dict, List, Union
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from textblob import TextBlob


class NLPTextProcessor:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.spacy_model = spacy.load("en_core_web_sm")

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text

    def tokenize_words(self, text: str) -> List[str]:
        return word_tokenize(text)

    def tokenize_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def get_pos_tags(self, tokens: List[str]) -> List[tuple]:
        return pos_tag(tokens)

    def get_named_entities_nltk(self, text: str) -> str:
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        entities = ne_chunk(tagged_tokens)
        return str(entities)

    def get_named_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        doc = self.spacy_model(text)
        entity_map = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": []
        }
        for entity in doc.ents:
            if entity.label_ in entity_map:
                entity_map[entity.label_].append(entity.text)
        return entity_map

    def get_nltk_sentiment(self, text: str) -> Dict[str, float]:
        return self.sentiment_analyzer.polarity_scores(text)

    def get_textblob_sentiment(self, text: str) -> Dict[str, float]:
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    def process(self, text: str) -> Dict[str, Union[str, List, Dict]]:
        cleaned = self.clean_text(text)
        tokens = self.tokenize_words(cleaned)
        tokens_no_stop = self.remove_stopwords(tokens)
        lemmas = self.apply_lemmatization(tokens)
        pos_tags = self.get_pos_tags(lemmas)
        sentences = self.tokenize_sentences(text)
        nltk_sentiment = self.get_nltk_sentiment(text)
        textblob_sentiment = self.get_textblob_sentiment(text)
        spacy_entities = self.get_named_entities_spacy(text)
        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "tokens_no_stopwords": tokens_no_stop,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "sentences": sentences,
            "nltk_sentiment": nltk_sentiment,
            "textblob_sentiment": textblob_sentiment,
            "named_entities": spacy_entities
        }


if __name__ == "__main__":
    processor = NLPTextProcessor()
    input_text = """Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and human language.
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California.
    I love NLP! It's amazing how computers can understand human language."""
    result = processor.process(input_text)
    print(result)
