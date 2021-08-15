import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_KEYS = [
    "pad_token",
    "unk_token",
]

SPECIAL_TOKENS_VALUES = [
    "[PAD]",
    "[UNK]",
]

SPECIAL_TOKENS = OrderedDict(zip(SPECIAL_TOKENS_KEYS, SPECIAL_TOKENS_VALUES))

class Vocab(object):
    def __init__(self, init_tokens = SPECIAL_TOKENS):
        self.init_tokens = init_tokens
        self.word2idx = {}
        self.idx2word = {}
        self.count = 0
        self._add_init_tokens()

    def _add_init_tokens(self):
        for token in self.init_tokens.values():
            self._add_word(token)

    def _add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.count
            self.idx2word[self.count] = word
            self.count += 1

    def add_words(self, words):
        for word in words:
            self._add_word(word)

