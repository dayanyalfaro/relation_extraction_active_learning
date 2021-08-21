import os
import spacy
import logging

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from itertools import permutations

from ann_scripts.anntools import Collection
from utils import save_pkl
from vocab import Vocab

logger = logging.getLogger(__name__)

logger.info('load spacy model...')
nlp = spacy.load('es_core_news_md')

rel2idx = {
            "none": 0,
            "in-place": 1,
            "has-property": 2,
            "target": 3,
            "part-of": 4,
            "arg": 5,
            "subject": 6,
            "entails": 7,
            "same-as":8,
            "in-context": 9,
            "in-time": 10,
            "causes": 11,
            "domain": 12,
            "is-a": 13
        }


def _add_tokens_index(data: List[Dict], vocab):
    unk_str = '[UNK]'
    unk_idx = vocab.word2idx[unk_str]

    for d in data:
        d['token2idx'] = [vocab.word2idx.get(i, unk_idx) for i in d['tokens']]

def _handle_pos_limit(pos: List[int], limit: int) -> List[int]:
    for i, p in enumerate(pos):
        if p > limit:
            pos[i] = limit
        if p < -limit:
            pos[i] = -limit
    return [p + limit + 1 for p in pos]

def _add_pos_seq(train_data: List[Dict], cfg):
        for d in train_data:
            d['head_pos'] = list(map(lambda i: i - d['head_start'], list(range(d['seq_len']))))
            d['head_pos'] = _handle_pos_limit(d['head_pos'], int(cfg.pos_limit))

            d['tail_pos'] = list(map(lambda i: i - d['tail_start'], list(range(d['seq_len']))))
            d['tail_pos'] = _handle_pos_limit(d['tail_pos'], int(cfg.pos_limit))


def _get_sentence_spans(sentence:str):
    return [(token.idx, token.idx+len(token)) for token in nlp(sentence)] 

def _get_pos_in_spans(start, end, spans):
    s = None
    e = None
    for i,span in enumerate(spans):
        e = i
        if not s and span[0] >= start:
            s = i
        if span[0] > end:
            e = i - 1
            return s, e
    return s, e

def _preprocess_collection(collection: Collection):
    features = []
    for sentence in tqdm(collection.sentences):
        sentences_relations = {(from_, to): 'none' for from_, to in permutations(sentence.keyphrases, 2)}

        sentence_spans = _get_sentence_spans(sentence.text)
        tokens = [token.text for token in nlp(sentence.text)]

        for relation in sentence.relations:
            sentences_relations[(relation.from_phrase,relation.to_phrase)] = relation.label

        for head, tail in permutations(sentence.keyphrases, 2):
            hs, he = _get_pos_in_spans(head.spans[0][0], head.spans[-1][1], sentence_spans)
            ts, te = _get_pos_in_spans(tail.spans[0][0], tail.spans[-1][1], sentence_spans)

            feature = {
                'sentence' : sentence.text,
                'tokens' : tokens,
                'seq_len' : len(tokens),
                'head_start' : hs,
                'head_end' : he,
                'tail_start' : ts,
                'tail_end' : te,
                'head_type' : head.label,
                'tail_type' : tail.label,
                'relation' : sentences_relations[head,tail],
                'rel2idx' : rel2idx[sentences_relations[head,tail]]
            }

            features.append(feature)
    return features


def preprocess(cfg):
    logger.info('=' * 10 + ' Start preprocess data ' + '=' * 10)

    train_path = Path(os.path.join(cfg.cwd, cfg.data_path, 'training'))
    valid_path = Path(os.path.join(cfg.cwd, cfg.data_path, 'develop'))
    test_path = Path(os.path.join(cfg.cwd, cfg.data_path, 'testing'))

    logger.info('load collections...')
    train_collection = Collection().load_dir(train_path)
    valid_collection = Collection().load_dir(valid_path)
    test_collection = Collection().load_dir(test_path)

    logger.info('process collections...')
    train_data = _preprocess_collection(train_collection)
    valid_data = _preprocess_collection(valid_collection)
    test_data = _preprocess_collection(test_collection)

    logger.info('build vocabulary...')
    vocab = Vocab()
    train_tokens = [d['tokens'] for d in train_data]
    valid_tokens = [d['tokens'] for d in valid_data]
    test_tokens = [d['tokens'] for d in test_data]
    sent_tokens = [*train_tokens, *valid_tokens, *test_tokens]
    for sent in sent_tokens:
        vocab.add_words(sent)

    logger.info('get index of tokens...')
    _add_tokens_index(train_data, vocab)
    _add_tokens_index(valid_data, vocab)
    _add_tokens_index(test_data, vocab)

    logger.info('build position sequence...')
    _add_pos_seq(train_data, cfg)
    _add_pos_seq(valid_data, cfg)
    _add_pos_seq(test_data, cfg)

    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, cfg.out_path), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'train.pkl')
    valid_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'valid.pkl')
    test_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)

    vocab_save_fp = os.path.join(cfg.cwd, cfg.out_path, 'vocab.pkl')
    vocab_txt = os.path.join(cfg.cwd, cfg.out_path, 'vocab.txt')
    save_pkl(vocab, vocab_save_fp)
    logger.info('save vocab in txt file, for watching...')
    with open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join(vocab.word2idx.keys()))

    logger.info('=' * 10 + ' End preprocess data ' + '=' * 10)







