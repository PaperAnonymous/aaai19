from __future__ import division

import sys
import time

import numpy as np
import os
import re
import zipfile
from collections import defaultdict
import gzip
import json
import pickle
import random
from matching import Jain_memory_tf
from memn2n import MemN2N
from simple_match import Simple_Match_Model
from kv_memn2n import KVMemN2N

def parse_QA_task(data_files, dic, update = True):
    question, answer = [], []

    for fp in data_files:
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                q2idx, a2idx = (), ()
                line = line.strip().lower()
                q_txt, a_txt = line.split("\t")

                q_txt = re.sub("'s", " 's", q_txt)
                q_txt = re.sub('(?<=\]),', '', q_txt)
                q_txt = q_txt.strip()
                q_txt = re.sub("[ ]+", " ", q_txt)
                q_txt = re.sub('[\[\]]', '', q_txt)
                q_txt = re.split('\_| ', q_txt)

                for i, w in enumerate(q_txt):
                    if w not in dic and update:
                        dic[w] = len(dic)
                    q2idx += (dic[w], )

                a_txt = re.sub("'s", " 's", a_txt)
                a_txt = re.sub("\_", " ", a_txt)
                a_txt = a_txt.split('|')
                for i , w in enumerate(a_txt):
                    a2idx += (w, )

                question += [q2idx]
                answer += [a2idx]

    return question, answer

def parse_freebase(data_files, dic, update = True):
    kb, sub_idx = np.zeros((30, 3, 300000), dtype = np.int32), defaultdict(list)
    max_line_idx, max_k = 0, 0

    for fp in data_files:
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                line = line.replace('\n', '').lower()
                words = line.split('|')

                for i in [0, 1, 2]:
                    sub = ()
                    word = words[i]

                    if i in [0, 2]:
                        word = re.sub("'s", " 's", word)
                        word = re.sub("[ ]+", " ", word)
                        word = word.split(' ')
                    elif i == 1:
                        word = re.sub("[ ]+", " ", word)
                        word = word.split('_')

                    for k, w in enumerate(word):
                        if w not in dic and update:
                            dic[w] = len(dic)
                        kb[k, i, line_idx] = dic[w]
                        sub += (dic[w], )

                    if i == 0:
                        sub_idx[sub] += [(1, line_idx)]
                    elif i == 2:
                        sub_idx[sub] += [(-1, line_idx)]

                    if max_k < k + 1:
                        max_k = k + 1
            if max_line_idx < line_idx + 1:
                max_line_idx = line_idx + 1

    return kb[:max_k, :, :max_line_idx], sub_idx

def idx2word(a, dic2):
    return (' ').join([ dic2[w] for w in a ])

def parse_freebase_memn2n(data_files, dic, update = True):
    kb, sub_idx = np.zeros((1, 3, 600000), dtype = np.int32), defaultdict(list)
    max_line_idx, max_k, line_idx = 0, 0, 0

    for fp in data_files:
        with open(fp) as f:
            for line in f:
                line = line.replace('\n', '').lower()
                words = line.split('|')

                for i in [0, 1, 2]:
                    sub = ()
                    word = words[i]

                    if i in [0, 2]:
                        word = re.sub("'s", " 's", word)

                    if i == 1:
                        word = re.sub('_', ' ', word)
                    if word not in dic and update:
                        dic[word] = len(dic)
                    kb[0, i, line_idx] = dic[word]

                    if i == 1:
                        word = word + ' -1'
                        if word not in dic and update:
                            dic[word] = len(dic)
                    if i == 0:
                        idx = 2
                    elif i == 2:
                        idx = 0
                    else:
                        idx = 1
                    kb[0, idx, line_idx + 1] = dic[word]

                    if i in [0, 2]:
                        word = word.split(' ')
                        for k, w in enumerate(word):
                            sub += (dic[w], )
                        if i == 0:
                            sub_idx[sub] += [(1, line_idx)]
                        elif i == 2:
                            sub_idx[sub] += [(1, line_idx + 1)]
                line_idx += 2

            if max_line_idx < line_idx:
                max_line_idx = line_idx

    return kb[:, :, :max_line_idx], sub_idx

def read_golden_rel(path, is_transform = False):
    goldens = []
    qtype2rel = {'actor_to_movie': 'starred actors -1',
    'director_to_movie': 'directed by -1', 'movie_to_actor': 'starred actors',
    'movie_to_director': 'directed by', 'movie_to_genre': 'has genre',
    'movie_to_language': 'in language', 'movie_to_tags': 'has tags',
    'movie_to_writer': 'written by', 'movie_to_year': 'release year',
    'tag_to_movie': 'has tags -1', 'writer_to_movie': 'written by -1',
    'movie_to_imdbrating': 'has_imdb_rating', 'movie_to_imdbvotes': 'has_imdb_votes'}
    with open(path) as f:
        for line in f:
            line = line.replace('\n', '').lower()
            golden = []
            if is_transform:
                line = line.split('_')
                for i in range(int(np.ceil(len(line)/3.))):
                    golden += [qtype2rel['_'.join(line[i*2: i*2+3])]]
                golden = ' '.join(golden)
            else:
                golden_num = int(len(re.findall('#', line))/2)
                line = '#'.join([w for i, w in enumerate(line.split('#')) if i not in [2, 4, 6]])
                golden = re.sub('[\_|\#]+', ' ', line)
            goldens += [golden]
    return goldens

def read_pool(path, dic2):
    sequence_pool = []
    with gzip.open(path) as f:
        for line in f:

            line = line.replace('\n', '')
            sequence = ()
            for w in line.split(' '):
                sequence += (dic2[w], )
            sequence_pool += [sequence]
    return sequence_pool

def read_topic(path, dic):
    topics = []
    for file in ['train', 'dev', 'test']:
        topic = []
        with open('%s/qa_%s_mid.txt' %(path, file)) as f:
            for line_idx, line in enumerate(f):
                line = line.replace('\n', '')

                t = []
                for entity in line.split('\t'):
                    t += [tuple([dic[w] for w in entity.split(' ')])]
                topic += [t]

        topics += [topic]
    return topics

def read_kbidx(path):
    entity2kbidx = defaultdict(set)
    with gzip.open(path) as f:
        for line in f:
            line = line.replace('\n', '')

            entity, idx = line.split('\t')
            entity = re.sub(' ', '_', entity)
            idx = set(idx.split('|'))
            idx = random.sample(idx, np.min([500, len(idx)]))
            idx = map(int, idx)
            entity2kbidx[entity] = list(idx)
    return entity2kbidx

def build_model_tf(general_config, dictionary, load_id, save_id, outfile = None):
    if os.path.isfile("%s/weights%s.npy"%(outfile, load_id)):
        weights = np.array(np.load('%s/weights%s.npy' %(outfile, load_id)))
        print(weights.shape)
        vocab = weights.astype(np.float32)
        print(vocab[10, :10])
    else:
        if general_config.init_vocab:
            vocab = initialize_vocab(dictionary, \
                "/home/yunshi/Word2vec/glove.840B.300d.zip", False)
            if not os.path.isfile("%s/weights%s.npy"%(outfile, save_id)):
                np.save('%s/weights%s' %(outfile, save_id), vocab)
        else:
            vocab = np.random.uniform(-0.1, 0.1, (len(dictionary), 300))

    #return Jain_memory_tf(general_config, vocab), None
    # return MemN2N(general_config, vocab), None
    return Simple_Match_Model(general_config, vocab), None
    #return KVMemN2N(general_config, vocab), None

def initialize_vocab(dictionary, glove_file, split_trick=False):
    vocab = np.random.uniform(-0.1, 0.1, (len(dictionary), 300))
    if split_trick:
        phrase_vocab = defaultdict(lambda: np.zeros((300), dtype = np.float32))
        phrase_vocab_num = defaultdict(int)
        split_dictionary = defaultdict(list)
        num_dictionary = []
        for phrase in dictionary:
            words = phrase.split(' ')
            if not re.search('[^0-9]', phrase):
                num_dictionary += [phrase]
            if len(words) > 0:
                for word in words:
                    split_dictionary[word] += [phrase]
    seen = 0
    gloves = zipfile.ZipFile(glove_file)
    for glove in gloves.infolist():
        with gloves.open(glove) as f:
            for line in f:
                if line != "":
                    splitline = line.split()
                    word = splitline[0]
                    embedding = splitline[1:]
                    if word in dictionary and len(embedding) == 300:
                        temp = np.array([float(val) for val in embedding])
                        vocab[dictionary[word], :] = temp/np.sqrt(np.sum(temp**2))
                        seen += 1
                    if split_trick:
                        if word in split_dictionary and len(embedding) == 300:
                            temp = np.array([float(val) for val in embedding])
                            for phrase in split_dictionary[word]:
                                phrase_vocab[dictionary[phrase]] += temp/np.sqrt(np.sum(temp**2))
                                phrase_vocab_num[dictionary[phrase]] += 1
    if split_trick:
        for phrase in phrase_vocab:
            vocab[phrase, :] += phrase_vocab[phrase]/phrase_vocab_num[phrase]
        sorted_num = ['first'] + sorted(num_dictionary) + ['today', 'last']
        for i in range(1, len(sorted_num)-1):
            vocab[dictionary[sorted_num[i]], :] = (vocab[dictionary[sorted_num[i-1]], :]+\
                vocab[dictionary[sorted_num[i]], :]+vocab[dictionary[sorted_num[i+1]], :])/3
    vocab[0, :]  = 0
    print("pretrained vocab %s among %s" %(seen, len(dictionary)))
    return vocab

def save_config(config, path):
    with open(path, 'w') as f:
        for key, value in config.__dict__.iteritems():
            f.write(u'{0}: {1}\n'.format(key, value))
