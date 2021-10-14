import os
import bz2
import wget
import torch
import spacy
import shutil
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
from typing import List, Dict
from itertools import permutations
from transformers import BertTokenizer
from gensim.models.word2vec import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

from ann_scripts.anntools import Collection
from utils import save_pkl
from vocab import Vocab

logger = logging.getLogger(__name__)

rel2idx = { 'ehealthKD2021':
                            {"none": 0,
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
                            "is-a": 13,
                    },
            'mimlre':
                    {
                        'no_relation':0,
                        'per:employee_of':1,
                        'per:countries_of_residence':2,
                        'org:city_of_headquarters':3,
                        'org:country_of_headquarters':4,
                        'per:stateorprovinces_of_residence':5,
                        'per:cities_of_residence':6,
                        'per:title':7,
                        'org:top_members/employees':8,
                        'org:member_of':9,
                        'org:parents':10,
                        'org:alternate_names':11,
                        'org:stateorprovince_of_headquarters':12,
                        'org:founded_by':13,
                        'per:country_of_birth':14,
                        'org:founded':15,
                        'per:city_of_birth':16,
                        'org:subsidiaries':17,
                        'per:spouse':18,
                        'per:origin':19,
                        'per:stateorprovince_of_birth':20,
                        'per:alternate_names':21,
                        'per:date_of_death':22,
                        'org:members':23,
                        'per:date_of_birth':24,
                        'per:children':25,
                        'per:parents':26,
                        'per:country_of_death':27,
                        'org:dissolved':28,
                        'per:city_of_death':29,
                        'org:political/religious_affiliation':30,
                        'per:age':31,
                        'per:schools_attended':32,
                        'per:stateorprovince_of_death':33,
                        'per:other_family':34,
                        'per:siblings':35,
                        'org:shareholders':36,
                        'per:cause_of_death':37,
                        'org:number_of_employees/members':38,
                        'per:religion':39,
                        'per:charges':40,
                    }
        }

class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, cfg):
        self.corpus = cfg.corpus.name
        self.download_link = cfg.corpus.download_link
        self.zip_path = cfg.corpus.zip_path
        self.path_word = cfg.cwd + cfg.corpus.pretrained_path  # path of pre-trained word embedding

    def load_embedding(self):
        if self.corpus == 'ehealthKD2021':
            if not os.path.exists(self.path_word):
                logger.info('Downloading the pretrained embedding')
                wget.download(self.download_link)
                vec_zip_path = self.zip_path
                logger.info('Extracting the pretrained embedding')

                with bz2.open(vec_zip_path, 'rb') as f_in:
                    with open(self.path_word, 'xb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            wv = KeyedVectors.load_word2vec_format(self.path_word)
        else:
            if not os.path.exists(self.path_word):
                logger.info('Downloading the pretrained embedding')
                wget.download(self.download_link)
                vec_zip_path = self.zip_path
                logger.info('Extracting the pretrained embedding')

                with ZipFile(vec_zip_path, 'r') as zip:
                    zip.extractall(self.path_word)

            glove_file = datapath(self.path_word + 'glove.6B.300d.txt')
            tmp_file = get_tmpfile(self.path_word + 'glove2word2vec.6B.300d.txt')
            _ = glove2word2vec(glove_file, tmp_file)
            wv = KeyedVectors.load_word2vec_format(tmp_file)

        word2idx = { w : k + 2 for (w,k) in wv.key_to_index.items()}  # word to wordID
        word2idx['[PAD]'] = 0  # PAD character
        word2idx['[UNK]'] = 1  # UNK character

        num_words, embedding_dim = wv.vectors.shape
        embedding_matrix = np.zeros((num_words + 2, embedding_dim))
        embedding_matrix[2:] = wv.vectors
        embedding_matrix = embedding_matrix.astype(np.float32)
        embedding_matrix = torch.from_numpy(embedding_matrix)

        return word2idx, embedding_matrix

def _encoder_serialize(data: List[Dict], cfg):
    """
    Implement the following input formats:
        - entity_mask: [SUBJ-NER], [OBJ-NER].
        - entity_marker: [E1] subject [/E1], [E2] object [/E2].
        - entity_marker_punct: @ subject @, # object #.
        - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
        - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
    """
    logger.info('use bert tokenizer...')
    
    input_format = cfg.model.input_format

    if cfg.model.offline:
        tokenizer = BertTokenizer.from_pretrained(cfg.cwd + '/embedding/bert-base-multilingual-cased/')
    else:
        tokenizer = BertTokenizer.from_pretrained(cfg.model.lm_file)

    new_tokens = []
    if input_format == 'entity_marker':
        new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        tokenizer.add_tokens(new_tokens)

    for d in data:
        sents = []
        tokens, subj_type, obj_type = d['tokens'], d['head_type'], d['tail_type']
        ss, se, ostart, oe =  d['head_start'], d['head_end'], d['tail_start'], d['tail_end']

        if input_format == 'entity_mask':
            subj_type = '[SUBJ-{}]'.format(subj_type)
            obj_type = '[OBJ-{}]'.format(obj_type)
            for token in (subj_type, obj_type):
                if token not in new_tokens:
                    new_tokens.append(token)
                    tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker':
            subj_start = '[SUBJ-{}]'.format(subj_type)
            subj_end = '[/SUBJ-{}]'.format(subj_type)
            obj_start = '[OBJ-{}]'.format(obj_type)
            obj_end = '[/OBJ-{}]'.format(obj_type)
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in new_tokens:
                    new_tokens.append(token)
                    tokenizer.add_tokens([token])
        elif input_format == 'typed_entity_marker_punct':
            subj_type = tokenizer.tokenize(subj_type.replace("_", " ").lower())
            obj_type = tokenizer.tokenize(obj_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = tokenizer.tokenize(token)

            if input_format == 'entity_mask':
                if ss <= i_t <= se or os <= i_t <= oe:
                    tokens_wordpiece = []
                    if i_t == ss:
                        new_ss = len(sents)
                        tokens_wordpiece = [subj_type]
                    if i_t == ostart:
                        new_os = len(sents)
                        tokens_wordpiece = [obj_type]

            elif input_format == 'entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['[E1]'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['[/E1]']
                if i_t == ostart:
                    new_os = len(sents)
                    tokens_wordpiece = ['[E2]'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['[/E2]']

            elif input_format == 'entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == ostart:
                    new_os = len(sents)
                    tokens_wordpiece = ['#'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['#']

            elif input_format == 'typed_entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [subj_start] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + [subj_end]
                if i_t == ostart:
                    new_os = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            elif input_format == 'typed_entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == ostart:
                    new_os = len(sents)
                    tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            sents.extend(tokens_wordpiece)

        # sents = sents[:cfg.model.max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        d['token2idx'] = input_ids[:512]
        d['seq_len'] = len(d['token2idx'])
        d['head_start'] =  new_ss + 1
        d['tail_start'] =  new_os + 1

def _lm_serialize(data: List[Dict], cfg):
    logger.info('use bert tokenizer...')
    if cfg.model.offline:
        tokenizer = BertTokenizer.from_pretrained(cfg.cwd + '/embedding/bert-base-multilingual-cased/')
    else:
        tokenizer = BertTokenizer.from_pretrained(cfg.model.lm_file)
    for d in data:
        sent = d['sentence'].strip()
        sent = sent.replace(d['head'], d['head_type'], 1).replace(d['tail'], d['tail_type'], 1)
        sent += '[SEP]' + d['head'] + '[SEP]' + d['tail']
        d['token2idx'] = tokenizer.encode(sent, add_special_tokens=True)[:512]
        d['seq_len'] = len(d['token2idx'])

def _add_tokens_index(data: List[Dict], word2idx):
    unk_str = '[UNK]'
    unk_idx = word2idx[unk_str]

    for d in data:
        d['token2idx'] = [word2idx.get(i, unk_idx) for i in d['tokens']]

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


def _get_sentence_spans(sentence:str, nlp):
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

def _preprocess_collection(collection: Collection,cfg, nlp):
    features = []
    for sentence in tqdm(collection.sentences):
        sentences_relations = {(from_, to): 'none' for from_, to in permutations(sentence.keyphrases, 2)}

        sentence_spans = _get_sentence_spans(sentence.text, nlp)
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
                'head' : head.text,
                'tail' : tail.text,
                'head_start' : hs,
                'head_end' : he,
                'tail_start' : ts,
                'tail_end' : te,
                'head_type' : head.label,
                'tail_type' : tail.label,
                'relation' : sentences_relations[head,tail],
                'rel2idx' : rel2idx[cfg.corpus.name][sentences_relations[head,tail]]
            }

            features.append(feature)
    return features

def _preprocess_dataframe(df,cfg, nlp):
    features = []
    for index, row in df.iterrows():
        if row['key'] == 'key':
            continue
        tokens = [token.text for token in nlp(row['sentence'])]
        entity = nlp(row['entity'].strip())[0].text
        hs = tokens.index(nlp(row['entity'].strip())[0].text)
        he = tokens.index(nlp(row['entity'].strip())[-1].text)
        ts = tokens.index(nlp(row['slotValue'].strip())[0].text)
        te = tokens.index(nlp(row['slotValue'].strip())[-1].text)

        feature = {
            'sentence' : row['sentence'],
            'tokens' : tokens,
            'seq_len' : len(tokens),
            'head' : row['entity'],
            'tail' : row['slotValue'],
            'head_start' : hs,
            'head_end' : he,
            'tail_start' : ts,
            'tail_end' : te,
            'head_type' : 'head',
            'tail_type' : 'tail',
            'relation' : row['relation'],
            'rel2idx' : rel2idx[cfg.corpus.name][row['relation']],
        }

        features.append(feature)
    return features

def preprocess(cfg):
    logger.info('=' * 10 + ' Start preprocess data ' + '=' * 10)

    logger.info('load spacy model...')
    if cfg.corpus.name == 'ehealthKD2021':
        nlp = spacy.load('es_core_news_md')
    else:
        nlp = spacy.load('en_core_web_md')

    if cfg.corpus.name == 'ehealthKD2021':
        train_path = Path(os.path.join(cfg.cwd, cfg.corpus.data_path, 'training'))
        valid_path = Path(os.path.join(cfg.cwd, cfg.corpus.data_path, 'develop'))
        test_path = Path(os.path.join(cfg.cwd, cfg.corpus.data_path, 'testing'))

        logger.info('load collections...')
        train_collection = Collection().load_dir(train_path)
        valid_collection = Collection().load_dir(valid_path)
        test_collection = Collection().load_dir(test_path)

        logger.info('process collections...')
        train_data = _preprocess_collection(train_collection,cfg, nlp)
        if not cfg.use_all_train:
            train_data = train_data[:cfg.train_size]
        valid_data = _preprocess_collection(valid_collection,cfg, nlp)
        test_data = _preprocess_collection(test_collection,cfg, nlp)
    else:
        logger.info('load and divide csv dataset...')
        csv_path = Path(os.path.join(cfg.cwd, cfg.corpus.data_path, 'annotated_sentences.csv'))
        df = pd.read_csv(csv_path) # .sample(frac = 0.01)
        train_df = df.sample(frac = 0.6, random_state = 1)
        rest_df = df.drop(train_df.index)
        valid_df = rest_df.sample(frac = 0.25, random_state = 1)
        test_df = rest_df.drop(valid_df.index)

        logger.info('process dataframes...')
        train_data = _preprocess_dataframe(train_df,cfg,nlp)
        valid_data = _preprocess_dataframe(valid_df,cfg,nlp)
        test_data = _preprocess_dataframe(test_df,cfg, nlp)

    if cfg.model.model_name == 'lm':
        logger.info('use pretrained language models serialize sentence...')
        _lm_serialize(train_data, cfg)
        _lm_serialize(valid_data, cfg)
        _lm_serialize(test_data, cfg)
    elif cfg.model.model_name == 'encoder':
        logger.info('use pretrained language model serialize sentence...')
        _encoder_serialize(train_data, cfg)
        _encoder_serialize(valid_data, cfg)
        _encoder_serialize(test_data, cfg)
    else:
        if cfg.corpus.use_pretrained:
            logger.info('load word embedding...')
            word2idx, word_vec = WordEmbeddingLoader(cfg).load_embedding()
            cfg.word_dim = word_vec.shape[1]

            logger.info('save word2vec file...')
            word_vec_save_fp = os.path.join(cfg.cwd, cfg.corpus.out_path, 'word2vec.pkl')
            save_pkl(word_vec, word_vec_save_fp)
        else:
            logger.info('build vocabulary...')
            vocab = Vocab()
            train_tokens = [d['tokens'] for d in train_data]
            valid_tokens = [d['tokens'] for d in valid_data]
            test_tokens = [d['tokens'] for d in test_data]
            sent_tokens = [*train_tokens, *valid_tokens, *test_tokens]
            for sent in sent_tokens:
                vocab.add_words(sent)

            word2idx = vocab.word2idx

            logger.info('save vocab file...')
            vocab_save_fp = os.path.join(cfg.cwd, cfg.corpus.out_path, 'vocab.pkl')
            vocab_txt = os.path.join(cfg.cwd, cfg.corpus.out_path, 'vocab.txt')
            save_pkl(vocab, vocab_save_fp)
            logger.info('save vocab in txt file, for watching...')
            with open(vocab_txt, 'w', encoding='utf-8') as f:
                f.write(os.linesep.join(vocab.word2idx.keys()))

        logger.info('get index of tokens...')
        _add_tokens_index(train_data, word2idx)
        _add_tokens_index(valid_data, word2idx)
        _add_tokens_index(test_data, word2idx)

        logger.info('build position sequence...')
        _add_pos_seq(train_data, cfg)
        _add_pos_seq(valid_data, cfg)
        _add_pos_seq(test_data, cfg)

    logger.info('save data for backup...')
    os.makedirs(os.path.join(cfg.cwd, cfg.corpus.out_path), exist_ok=True)
    train_save_fp = os.path.join(cfg.cwd, cfg.corpus.out_path, 'train.pkl')
    valid_save_fp = os.path.join(cfg.cwd, cfg.corpus.out_path, 'valid.pkl')
    test_save_fp = os.path.join(cfg.cwd, cfg.corpus.out_path, 'test.pkl')
    save_pkl(train_data, train_save_fp)
    save_pkl(valid_data, valid_save_fp)
    save_pkl(test_data, test_save_fp)

    logger.info('=' * 10 + ' End preprocess data ' + '=' * 10)







