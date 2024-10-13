# Import tokenizers, data visualizers, data containers
import spacy
from spacy.tokens.token import Token as SpacyToken
from conllu.models import Token as ConlluToken
from conllu.models import TokenList

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

from dataclasses import dataclass
from typing import List, Tuple, Dict


spacy_model = spacy.load("fr_core_news_sm")


@dataclass
class TokenPair:
    conllu_token: ConlluToken
    spacy_token: SpacyToken


class Sentence:
    def __init__(self, token_list: TokenList, model=spacy_model):
        self.text = token_list.metadata['text']
        conllu_tokens = [token for token in token_list]
        doc = model(self.text)
        model_tokens = [token for token in doc]
        self.token_pairs = self.__create_token_pairs(conllu_tokens, model_tokens)
        self.length = len(conllu_tokens)
        self.length_token_pairs = len(self.token_pairs)
        self.wer_count = self.__count_wer()

    def __repr__(self):
        return self.text

    def __create_token_pairs(self, conllu_tokens, model_tokens) -> List[TokenPair]:
        # Pair each conllu token with the first model token having the same form
        token_pairs = []
        for conllu_token in conllu_tokens:
            for model_token in model_tokens:
                if conllu_token["form"] == model_token.text:
                    token_pair = TokenPair(conllu_token, model_token)
                    token_pairs.append(token_pair)
                    model_tokens.remove(model_token)
                    break
        return token_pairs

    def __count_wer(self) -> Tuple[int, int]:
        # Return the correct number the total number of token pairs
        total_number = self.length_token_pairs
        correct_number = 0

        for pair in self.token_pairs:
            if pair.conllu_token["upos"] == pair.spacy_token.pos_:
                correct_number += 1

        return correct_number, total_number

    def get_label_count(self) -> Dict[str, int]:
        """Return {upos: number}"""
        label_count = {}
        for token_pair in self.token_pairs:
            conllu_token_upos = token_pair.conllu_token["upos"]
            label_count[conllu_token_upos] = label_count.get(conllu_token_upos, 0) + 1
        return label_count

    def compute_accuracy(self) -> float:
        """Get accuracy of current sentence"""
        correct, total = self.wer_count
        try:
            return correct / total
        except ZeroDivisionError:
            return 0.0


class Corpus:
    def __init__(self):
        self.sentences: List[Sentence] = []

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, key):
        if isinstance(key, slice):
            new_corpus = Corpus()
            new_corpus.sentences = self.sentences[key]
            return new_corpus
        else:
            return self.sentences[key]

    def __iter__(self):
        return iter(self.sentences)

    def copy(self):
        corpus_copied = Corpus()
        corpus_copied.sentences = self.sentences
        return corpus_copied

    def append(self, sentence: Sentence) -> None:
        if not isinstance(sentence, Sentence):
            raise TypeError("Can only append Sentence to Corpus.")
        self.sentences.append(sentence)

    def sort(self, reverse: bool) -> None:
        # Sort by sentence length
        self.sentences = sorted(self.sentences, key=lambda sentence: sentence.length, reverse=reverse)

    def get_label_count(self, sort=False, reverse=False) -> Dict[str, int]:
        # Get corpus' upos count
        corpus_label_count = {}
        for sentence in self.sentences:
            sentence_label_count = sentence.get_label_count()
            for label, count in sentence_label_count.items():
                corpus_label_count[label] = corpus_label_count.get(label, 0) + count

        if sort:
            corpus_sorted_tuple = sorted(corpus_label_count.items(), key=lambda x: x[1], reverse=reverse)
            corpus_label_count = dict(corpus_sorted_tuple)

        return corpus_label_count

    def compute_wer(self) -> float:
        """Corpus word error rate"""
        correct_corpus, total_corpus = 0, 0
        for sentence in self.sentences:
            correct_sentence, total_sentence = sentence.wer_count[0], sentence.wer_count[1]
            correct_corpus += correct_sentence
            total_corpus += total_sentence
        return correct_corpus / total_corpus

    def compute_ser(self) -> float:
        """Sentence error rate"""
        correct_sentence = 0
        for sentence in self.sentences:
            if sentence.compute_accuracy() == 1.0:
                correct_sentence += 1

        return correct_sentence / len(self)

    def __get_label_list(self) -> Tuple[list, list]:
        """Get true label list and prediction label list"""
        true_labels = []
        predict_labels = []
        for sentence in self.sentences:
            true_labels_sentence = [token.conllu_token["upos"] for token in sentence.token_pairs]
            predict_labels_sentence = [token.spacy_token.pos_ for token in sentence.token_pairs]
            true_labels.extend(true_labels_sentence)
            predict_labels.extend(predict_labels_sentence)
        return true_labels, predict_labels

    def compute_f1(self, option: str) -> float:
        if option not in ['micro', 'macro', 'weighted']:
            raise ValueError("Valid options: micro, macro, weighted.")

        true_labels, predict_labels = self.__get_label_list()
        return f1_score(true_labels, predict_labels, average=option)
