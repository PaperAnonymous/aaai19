# uncompyle6 version 3.0.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.14 |Anaconda custom (64-bit)| (default, Dec  7 2017, 17:05:42)
# [GCC 7.2.0]
# Embedded file name: /home/yunshi/RationaleNeuralPredictions/my_rationale/baseline/train_test_MetaQA.py
# Compiled at: 2018-03-12 22:17:31
from __future__ import division
import math, sys, numpy as np
from multiprocessing import Queue
from collections import defaultdict
import multiprocessing, threading, operator, os, collections, tensorflow as tf, time, json, random, re, glob, gzip
from util import *
import editdistance

np.random.seed(123)
random.seed(123)
train_story = None
alias = None
subject_idx = None
obj_dic = None
object_idx = None
mode, task = (None, None)
story_h5py = None
np.set_printoptions(suppress=True, threshold=np.nan)
#stop_words = ('writer film actors director').split(' ')
global stop_words
stop_words = set(['the','of','by','movies','films','in','who','what',
    'directed','which','with','are','that','were','director','a','actors',
    'also','movie','stared','film','is','was','acted','writer','directors',
    'released','and','screen','writers','share','languages','wriden','acters',
    'same','when','for','to','did','shared','years','whose','actor','as',
    'person','rote','listed','starred','appear','movi','wridden','start',
    'do','where','have',"'s",'fallunder','ere','spoken','release','types',
    'chair','pho','whathoneres','','rode','green','his','ind','wrote','their',
    'together','relise','man','an','ar','primary','words','it','on','about',
    'can','states','releese','i','at','co','directer','described','does',
    'yers','wrider','be','riter','whos','wended','this','you','right','state',
    'gender', 'child'])
qtype2rel = ['starred actors', 'directed by', 'has genre',
    'in language', 'has tags', 'written by', 'release year',
    'has_imdb_rating', 'has_imdb_votes']

def id(x):
    return x.__array_interface__['data'][0]

def train(story, sub_idx2, train_question, train_qstory, model, general_config, dic, load, rel=None):
    global dictionary
    global dictionary2
    global outfile
    global kb
    kb = story
    global sub_idx
    global stop_words
    sub_idx = sub_idx2
    train_config = general_config.train_config
    nepochs = general_config.nepochs
    batch_size = 64
    hop = general_config.train_config['max_hop']
    outfile = general_config.outfile
    lrate = 0.01#train_config['init_lrate']
    candidates_num = train_config['candidates_num']
    max_words = 30#train_config['max_words']
    train_questions = (
     train_question[0], train_qstory[0])
    valid_questions = (train_question[1], train_qstory[1])
    test_questions = (train_question[2], train_qstory[2])
    dictionary = dic
    task, in_dim, dropout, load, only_eval, add_sib, load_id, is_train, is_post_processing, mode, tid = load
    dictionary2 = {}
    for d in dictionary:
        dictionary2[dictionary[d]] = d
    if isinstance(list(stop_words)[0], str):
        stop_words = set([dic[w] for w in stop_words if w in dic])
    global sub_ngram
    sub_ngram = None#subidx2ngrams(sub_idx, dictionary, dictionary2, n=3)
    # print(sub_ngram.keys()[:100])

    if hop == 3 and train_config['model'] == 'matching':
        print('matching')
        sequence_pool = read_pool('./data/3-hop/seq_kb.txt.gz', dictionary)
        obj_pool = read_pool('./data/3-hop/obj_kb.txt.gz', dictionary)
        entity2sequence = pickle.load(open('./data/3-hop/entity2sequence.pkl', 'r'))
        sequence2obj = pickle.load(open('./data/3-hop/sequence2obj.pkl', 'r'))
        topic = read_topic('./data/mix-hop/audio', dictionary)
    elif hop == 3:
        entity2kbidx = read_kbidx('./data/mix-hop/entity2kbidx.txt.gz')
        topic = read_topic('./data/mix-hop/audio', dictionary)
        #global entity
        #entity = set(entity2kbidx.keys())
        #print entity2kbidx.keys()[:10]

    min_val_error, min_test_error = -np.inf, -np.inf

    try:
        min_test_error = find_max_test(outfile, tid)
    except:
        print('min test error start from -inf ...')

    print(model.obtain_var())
    saver = tf.train.Saver(get_weights_and_biases())
    if load:
        try:
            saver.restore(model._sess, '%s/model%s.ckpt' % (outfile, load_id))
            print('successfully load ...')
            print(model.obtain_var())
        except:
            print('fail to load pre-trained parameters ...')

    print(model.obtain_embedding()[10, :10])
    print('finish preparing ...')
    time1 = time.time()
    for ep in range(nepochs):
        total_err = 0.0
        total_cost = 0.0
        total_num = 0.0001

        if is_train:# batch_size
            dyn_batch = create_batches(len(train_questions[0]), batch_size)

        batch_num = len(dyn_batch) if is_train else 0
        idx = 0
        preds, batches, evals, top_candidates, top_probs = ([], [], [], [], [])

        for k in range(batch_num): #
            print('training %s ....' % idx)
            idx += 1
            batch = dyn_batch[k]

            if hop == 3 and train_config['model'] == 'matching':
                input_data, target_hop, memory_data, memory_obj, memory_sib \
                    = obtain_xys(train_questions, sequence_pool, obj_pool,
                    batch, dictionary2, hop, max_words=max_words,
                    entity2sequence=entity2sequence, sequence2obj=sequence2obj,
                    topic=topic[0])
            else:
                if hop == 3 and train_config['model'] == 'memn2n':
                    input_data, target_hop, memory_data, memory_obj, memory_sib\
                     = obtain_xys(train_questions, story, None, batch, dictionary2,
                        hop, max_words=max_words, entity2sequence=entity2kbidx,
                    topic=topic[0])
                else:
                    input_data, target_hop, memory_data, memory_obj, memory_sib\
                     = obtain_xys_thread(train_questions, story, sub_idx, batch,
                        dictionary2, 3, max_words=max_words, bt = rel[0])

            print('start propogation ...')
            time1 = time.time()
            print(batch)
            print('question %s\tmemory data %s \t sib %s \t target %s' % (
             str(input_data.T.shape), str(memory_data.shape), str(memory_sib.shape),
             str(target_hop.shape)))

            cost, probs, show, show1 = model.batch_rel_fit(memory_data, memory_sib,
                input_data, target_hop, dropout, lrate)
            print(probs[0, :10])
            total_cost += cost

            print('cost %s' % cost)
            print('propogation ... %s' % (time.time() - time1))
            time1 = time.time()

            golden = rel[0] if train_config['eval'] == 'rel_acc' else train_questions[1]
            err, pred = get_F1(probs, memory_data, golden, memory_obj, dictionary2, batch)

            print('cor %s' % np.sum(err))

            total_err += np.sum(err)
            total_num += len(batch)
            preds += pred
            batches += list(batch)
            evals += err

            for b in range(len(batch)):
                print('index: %s \t generate num: %s \t eval: %s\ttarget: %s' % (
                 batch[b], len(memory_obj[b]), err[b], np.sum(target_hop[b])))

            if idx % 10 == 0:
                model.save_model(saver, outfile, tid)
            print(model.obtain_var())

        train_error = total_err / total_num
        total_cost /= total_num

        message = '%d | train loss %g | train eval: %g ' % (
         ep + 1, total_cost, train_error)
        print('validation ...')

        total_val_err, add = (0.0, 0.0)
        total_val_num = 0
        preds, batches, evals, top_candidates, top_probs = ([], [], [], [], [])

        if only_eval:
            dyn_batch = [[11791]]
        else:
            dyn_batch = create_batches(len(valid_questions[0]), batch_size)

        for k in range(len(dyn_batch)):#
            batch = dyn_batch[k]

            if hop == 3 and train_config['model'] == 'matching':
                input_data, target_hop, memory_data, memory_obj, memory_sib \
                = obtain_xys(valid_questions, sequence_pool, obj_pool, batch,
                dictionary2, hop, max_words=max_words,
                entity2sequence=entity2sequence, sequence2obj=sequence2obj,
                topic = topic[1])
            else:
                if hop == 3 and train_config['model'] == 'memn2n':
                    input_data, target_hop, memory_data, memory_obj, memory_sib \
                    = obtain_xys(valid_questions, story, None, batch, dictionary2,
                        hop, max_words=max_words, entity2sequence=entity2kbidx,
                        topic = topic[1])
                else:
                    input_data, target_hop, memory_data, memory_obj, memory_sib\
                     = obtain_xys_thread(valid_questions, story, sub_idx,
                        batch, dictionary2, 3, max_words=max_words, bt = rel[1])

            print('data shape %s\tsib shape %s' % (memory_data.shape, memory_sib.shape))

            probs, show, show1 = model.predict_rel(memory_data, memory_sib, input_data, target_hop)

            golden = rel[1] if train_config['eval'] == 'rel_acc' else valid_questions[1]
            err, pred = get_F1(probs, memory_data, golden, memory_obj, dictionary2, batch)
            embedding = model.obtain_embedding()

            for b in range(len(batch)):
                if only_eval:
                    print('>>> Q: %s\t>>> A:%s' % (idx2word(input_data[b, :], dictionary2),
                     valid_questions[1][dyn_batch[k][b]]))
                    for j in range(len(memory_obj[b])):
                        print('%s\t%s' % (str([ dictionary2[w] for w in memory_data[b, j, :] ]), probs[(b, j)]))
                        if len(memory_obj[b][j]) < 50:
                            print([ idx2word(w, dictionary2) for w in memory_obj[b][j] ])
                        print('*****************')

                    stop

                print('index %s\tgenerate num %s\teval %s\ttarget %s' % (
                 batch[b], len(memory_obj[b]), err[b], np.sum(target_hop[b])))

            total_val_err += np.sum(err)
            preds += pred
            batches += list(batch)
            evals += err
            total_val_num += len(batch)

        val_error = total_val_err / np.max([total_val_num, 1e-10])
        val_error2 = (total_val_err + add) / np.max([total_val_num, 1e-10])
        message += ' | val eval: %g(%g)' % (val_error, val_error2)
        pred = print_pred(preds, batches, evals)

        print('test ...')
        test_error = 0
        if val_error2 > -1:
            if not only_eval:
                f = open('%s/pred%s_top1.txt' % (outfile, tid), 'w')
                f.write('\n'.join(pred))
                f.close()

            total_test_err, add = (0.0, 0.0)
            total_test_num = 0
            preds, batches, evals, top_candidates, top_probs = ([], [], [], [], [])
            dyn_batch = create_batches(len(test_questions[0]), batch_size)

            for k in range(len(dyn_batch)): #
                batch = dyn_batch[k]

                if hop == 3 and train_config['model'] == 'matching':
                    input_data, target_hop, memory_data, memory_obj, memory_sib\
                     = obtain_xys(test_questions, sequence_pool, obj_pool,
                        batch, dictionary2, hop, max_words=max_words,
                        entity2sequence=entity2sequence, sequence2obj=sequence2obj,
                        topic = topic[2])
                else:
                    if hop == 3 and train_config['model'] == 'memn2n':
                        input_data, target_hop, memory_data, memory_obj, memory_sib\
                         = obtain_xys(test_questions, story, None, batch, dictionary2,
                            hop, max_words=max_words, entity2sequence=entity2kbidx,
                            topic = topic[2])
                    else:
                        input_data, target_hop, memory_data, memory_obj, memory_sib\
                         = obtain_xys_thread(test_questions, story, sub_idx, batch,
                            dictionary2, 3, max_words=max_words, bt = rel[2])

                print('idx %s\tdata shape %s\tsib shape %s' %\
                    (k, memory_data.shape, memory_sib.shape))

                probs, show, show1 = model.predict_rel(memory_data, memory_sib,
                    input_data, target_hop)

                golden = rel[2] if train_config['eval'] == 'rel_acc' else test_questions[1]
                err, pred = \
                    get_F1(probs, memory_data, golden, memory_obj, dictionary2, batch)

                for b in range(len(batch)):
                    print('index %s\tgenerate num %s\teval %s\ttarget %s' % (
                     batch[b], len(memory_obj[b]), err[b], np.sum(target_hop[b])))

                total_test_err += np.sum(err)
                preds += pred
                batches += list(batch)
                evals += err
                total_test_num += len(batch)

            test_error = total_test_err / np.max([total_test_num, 1e-10])
            test_error2 = (total_test_err + add) / np.max([total_test_num, 1e-10])
            message += ' | Test eval: %f(%f)' % (test_error, test_error2)
            if test_error > min_test_error:
                min_test_error = test_error
                pred = print_pred(preds, batches, evals)
                f = open('%s/pred%s_top1_test.txt' % (outfile, tid), 'w')
                f.write(('\n').join(pred))
                f.close()
        print(message)
        if is_train:
            if test_error == 0 and val_error > min_val_error or test_error > 0 and min_test_error == test_error:
                model.save_model(saver, outfile, tid)
                min_val_error = val_error
                message += ' (saved-model) '
        else:
            stop
        log = open('%s/result%s.txt' % (outfile, tid), 'a')
        log.write(message + '\n')
        log.close()


def find_max_test(outfile, save_id):
    f = open('%s/result%s.txt' % (outfile, save_id), 'rb')
    max_test = -np.inf
    for line in f.readlines():
        line = line.replace('\n', '').strip().split(' ')
        test = float(re.findall('\\([0-9.]+\\)', line[-2])[0][1:-2])
        if max_test < test:
            max_test = test

    f.close()
    return max_test


def naive_get_F1(pred, ans):
    precision = len(set(pred) & set(ans)) * 1.0 / len(set(pred))
    recall = len(set(pred) & set(ans)) * 1.0 / len(set(ans))
    f1 = 2 * precision * recall / np.max([precision + recall, 1e-10])
    return f1


def obtain_xys(data, kb2, sub_idx2, batches, dic2, hop, max_words=None,
    entity2sequence=None, sequence2obj=None, topic=None):
    que, ans = data
    x = np.zeros((len(batches), 30), dtype=np.int32)
    y = np.zeros((len(batches), 10000))
    s = np.zeros((len(batches), 10000, 30), dtype=np.int32)
    sib = np.zeros((len(batches), 10000, 30), dtype=np.int32)
    o = []
    max_cand, max_xlen, max_slen, max_siblen = (0, 0, 0, 0)

    for i, b in enumerate(batches):
        time1 = time.time()
        story, obj, _ = obtain_story2(topic[b], kb2, sub_idx2, entity2sequence,
            sequence2obj, dic2)
        story1, obj1, _ = obtain_story(topic[b], kb, sub_idx, 2, dic2, is_con=False)
        story = story + story1
        obj = obj + obj1
        # story1, obj1, _ = obtain_story(que[b], kb, sub_idx, 1, dic2, is_con=True)
        # story = story + story1
        # obj = obj + obj1
        #print('obtaining story time ... %s' %(time.time() - time1))

        q = que[b]
        x[i, :len(q)] = q
        if max_xlen < len(q):
            max_xlen = len(q)
        story = story[:np.min([500, len(story)])]
        obj = obj[:np.min([500, len(story)])]
        for j in range(len(story)):
            s[i, j, :len(story[j])] = story[j]
            if max_slen < len(story[j]):
                max_slen = len(story[j])
            y[(i, j)] = naive_get_F1([ idx2word(w, dic2) for w in obj[j] ], ans[b])
            if sequence2obj is None:
                obj[j] = obj[j][:np.min([30, len(obj[j])])]
                for k in range(len(obj[j])):
                    sib[(i, j, k)] = obj[j][k][0]
                    if max_siblen < len(obj[j]):
                        max_siblen = len(obj[j])

        o += [obj]
        if max_cand < len(story):
            max_cand = len(story)
        y[i, :] = y[i, :] / np.max([np.sum(y[i, :]), 1e-10])

    x = x[:, :max_xlen]
    y = y[:, :max_cand]
    s = s[:, :max_cand, :max_slen]
    sib = sib[:, :max_cand, :max_siblen]
    return (x, y, s, o, sib)


def print_tmp(b, story, obj, ij, out):
    out.write('***%s\n' % b)
    for idx in range(len(story)):
        out.write('%s\t%s\t' % ((' ').join(map(str, ij[idx])),
            (' ').join(map(str, story[idx]))))
        obj_idx = []
        for jdx in range(len(obj[idx])):
            obj_idx += [(' ').join(map(str, list(obj[idx])[jdx]))]

        out.write('%s\n' % ('|').join(obj_idx))


def print_tmp2(b, story, obj, ij, out):
    out.write('***%s\n' % b)
    for idx in range(len(story)):
        out.write('%s\t%s\n' % (ij[idx],
            (' ').join(map(str, story[idx]))))


def obtain_xys_thread(data, kb, sub_idx, batches, dic2, hop, max_words=30, bt = None):

    def worker(batches, i, que, kb, sub_idx, hop, dic2):
        out = []
        #out = gzip.open('%s/tmp/test%s.txt.gz' % (outfile, i), 'wb')
        for idx, b in enumerate(batches):
            story, obj, head, ij = obtain_story(que[b], kb, sub_idx, 3, dic2, is_con=True)
            #out.write('%s\t%s\n' %(b, '\t'.join(map(str, head))))
            out += [(b, story, obj, ij)]
            #q.put((b, story, obj, ij))
            #print_tmp2(b, story, obj, ij, out)
            if idx % 500 == 0:
                print('%s\t%s' %(i, idx*1./len(batches)))
        #out.close()
        pickle.dump(out, open('%s/tmp/tmp%s.pkl' % (outfile, i), 'wb'), protocol=2)

    que, ans = data
    x = np.zeros((len(batches), max_words), dtype=np.int32)
    y = np.zeros((len(batches), 1000))
    s = np.zeros((len(batches), 1000, max_words), dtype=np.int32)
    sib = np.zeros((len(batches), 1000, 100), dtype=np.int32)
    o = []
    is_sib = False
    max_cand, max_xlen, max_slen, max_siblen = (0, 0, 0, 1)
    nprocs = 10
    chunksize = int(np.ceil(len(batches) / float(nprocs)))
    procs, idx_stream, story_stream, obj_stream, ij_stream = ([], [], [], [], [])
    if not os.path.exists('%s/tmp' % outfile):
        os.mkdir('%s/tmp' % outfile)
    time1 = time.time()
    for i in range(nprocs):
        start = chunksize * i
        end = len(batches) if i == (nprocs-1) else chunksize * (i + 1)
        p = multiprocessing.Process(target=worker, args=(
         batches[start: end], i,
         que, kb, sub_idx, hop, dic2))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    for f in glob.glob('%s/tmp/*.pkl' % outfile):
        out = pickle.load(open(f, 'rb'))
        for out_b in out:
            idx_stream += [out_b[0]]
            story_stream += [out_b[1]]
            obj_stream += [out_b[2]]
            ij_stream += [out_b[3]]
        os.remove(f)

    print('finish obtain ... ')
    for i, b in enumerate(batches):
        idx = idx_stream.index(b)
        story, obj, ij = story_stream[idx], obj_stream[idx], ij_stream[idx]
        q = que[b]
        x[i, :len(q)] = q
        if max_xlen < len(q):
            max_xlen = len(q)
        #print(idx2word(q, dic2))
        for j in range(np.min([300, len(story)])):
            s[i, j, :len(story[j])] = story[j]
            if max_slen < len(story[j]):
                max_slen = len(story[j])
            my_pred = idx2word(story[j], dic2)
            my_pred = re.sub('parents -1', 'children', my_pred)
            my_pred = re.sub('children -1', 'parents', my_pred)
            my_pred = re.sub('spouse -1', 'spouse', my_pred)
            my_pred = re.sub('-1', 'inverse', my_pred)
            if len(re.findall('|'.join(qtype2rel), bt[b])) == len(re.findall('|'.join(qtype2rel), my_pred)):
                y[i, j] = naive_get_F1([ idx2word(w, dic2) for w in obj[j]], ans[b])
            #print(idx2word(story[j], dic2))
            #print('%s\t%s' %(str([idx2word(w, dic2) for w in obj[j]]), y[i, j]))
            if is_sib:
                for k in range(np.min([50, len(obj[j])])):
                    sib[(i, j, k)] = obj[j][k][0]
                    if max_siblen < len(obj[j]):
                        max_siblen = len(obj[j])
        #exit()

        o += [obj[:300]]
        if max_cand < np.min([300, len(story)]):
            max_cand = np.min([300, len(story)])
        #y[i, :] = (y[i, :] >= np.max(y[i, :]))
        #y[i, :] = y[i, :] / np.max([np.sum(y[i, :]), 1e-10])

    x = x[:, :max_xlen]
    y = y[:, :max_cand]
    s = s[:, :max_cand, :max_slen]
    sib = sib[:, :max_cand, :max_siblen]
    return (x, y, s, o, sib)


def create_batches(N, batch_size, skip_idx=None):
    batches = []
    shuffle_batch = np.arange(N)
    if skip_idx:
        shuffle_batch = list(set(shuffle_batch) - set(skip_idx))
    np.random.shuffle(shuffle_batch)
    M = int((N - 1) / batch_size + 1)
    for i in range(M):
        batches += [shuffle_batch[i * batch_size:(i + 1) * batch_size]]

    return batches


def create_batches2(file, batch_limit, batch_size):
    idx = sorted(range(len(idx_num)), key=lambda k: idx_num[k])[::-1]
    batches, max_batch, num, dyn_batch, batch_num = ([], idx_num[idx[0]], 0, [], 0)
    for i in range(len(idx)):
        batch_num += max_batch
        num += 1
        batches += [idx[i]]
        if batch_num >= batch_limit or num >= batch_size:
            dyn_batch += [batches]
            batch_num, num = (0, 0)
            max_batch = idx_num[idx[i]]
            batches = []
        elif i + 1 == len(idx):
            dyn_batch += [batches]

    np.random.shuffle(dyn_batch)
    return dyn_batch


def get_weights_and_biases():
    """
    Return all weight and bias variables
    :return:
    """
    return [ var for var in tf.global_variables() if 'dic_A' not in var.name]


def obtain_story2(que, sequence_pool, obj_pool, entity2sequence, sequence2obj, dic2):
    ij = None
    if sequence2obj is not None:
        idx2que = defaultdict(set)
        idx_set = set()
        if not sub_ngram:
            if isinstance(que[0], int):
                for i in range(len(que)):
                    for j in range(i + 1, len(que) + 1):
                        if idx2word(que[i:j], dic2) in entity2sequence:
                            idx_set = idx_set | entity2sequence[idx2word(que[i:j], dic2)]
            else:
                for i in range(len(que)):
                    if idx2word(que[i], dic2) in entity2sequence:
                        idx_set = idx_set | entity2sequence[idx2word(que[i], dic2)]
        else:
            sub_dis = defaultdict(int)
            subs = set()
            for i in range(len(que)):
                for j in range(i+1, np.min([i+5, len(que)+1])):
                    if not set(que[i:j]) < set(stop_words):
                        #print(idx2word(que[i:j], dic2))
                        que_str = idx2word(que[i:j], dic2)
                        que_ngrams = ngrams(que_str, 3)
                        for que_ngram in que_ngrams:
                            subs = subs.union(sub_ngram[que_ngram])
                        for sub in subs:
                            sub_str = idx2word(sub, dic2)
                            dis = editdistance.eval(sub_str, que_str)
                            dis = 1. - dis*1./np.max([len(sub_str), len(que_str)])
                            if sub_dis[sub] < dis:
                                sub_dis[sub] = dis
            subs = sorted(sub_dis.items(), key=lambda v: v[1],
                reverse=True)[:5]
            for sub, _ in subs:
                if idx2word(sub, dic2) in entity2sequence:
                    idx_set = idx_set | entity2sequence[idx2word(sub, dic2)]
            ij = [sub for sub, _ in subs]

        story, objs = [], []
        for sidx in idx_set:
            story += [sequence_pool[sidx]]
            obj = []
            for oidx in sequence2obj[sidx]:
                obj += [obj_pool[oidx]]

            objs += [obj]

    else:
        idx2que = defaultdict(set)
        idx_set = set()
        if isinstance(que[0], int):
            for i in range(len(que)):
                for j in range(i + 1, len(que) + 1):
                    if idx2word(que[i:j], dic2) in entity2sequence:
                        idx_set = idx_set | set(entity2sequence[idx2word(que[i:j], dic2)])
        else:
            for i in range(len(que)):
                if idx2word(que[i], dic2) in entity2sequence:
                    idx_set = idx_set | set(entity2sequence[idx2word(que[i], dic2)])

        story, objs = [], defaultdict(set)
        seen = set()
        for sidx in idx_set:
            if sidx > 0:
                direction, c = 1, np.abs(sidx) - 1
            else:
                direction, c = -1, np.abs(sidx) - 1
            s, o, h = squeeze_story(sequence_pool[:, :, c], direction, True)
            if s not in seen:
                story += [s]
            objs[s].add(o)

        objs = [list(objs[s]) for s in story ]
    return (story, objs, ij)


def squeeze_story(a, direction, with_sub=True):
    if direction == 1:
        if with_sub:
            story = array2tuple(a[:, 0]) + array2tuple(a[:, 1])
        else:
            story = array2tuple(a[:, 1])
        head = array2tuple(a[:, 0])
        obj = array2tuple(a[:, 2])
    elif direction == -1:
        if with_sub:
            story = array2tuple(a[:, 2]) + array2tuple(a[:, 1]) + (1, )
        else:
            story = array2tuple(a[:, 1]) + (1, )
        head = array2tuple(a[:, 2])
        obj = array2tuple(a[:, 0])
    return (story, obj, head)

def ngrams(x, n):
    y = []
    x = re.sub('[^a-z]+', '', x)
    for i in range(len(x)-n+1):
        y += [x[i: i+n]]
    return y

def subidx2ngrams(sub_idx, dic, dic2, n=2):
    sub_ngram = defaultdict(set)
    for sub in sub_idx:
        sub_str = idx2word(sub, dic2)
        for ngram in ngrams(sub_str, n):
            sub_ngram[ngram].add(sub)
    #print([(w, sub_ngram[w]) for i, w in enumerate(sub_ngram) if i < 10])
    return sub_ngram

def obtain_story(que, kb, sub_idx, hop, dic2, is_con=True):

    def is_sublist(a, b):
        if a in b:
            return True

    def search_path(que, sub_idx, kb, hop, is_con):
        if hop == 1:
            #print(sub_ngram.keys()[:100])
            head, story, obj, candidate = ([], [], [], [])
            sub_i, sub_j = (0, 0)
            if isinstance(que[0], int):
                #print(idx2word(que, dic2))
                if not sub_ngram:
                    for i in range(len(que)):
                        for j in range(i + 1, len(que) + 1):
                            #print('%s\t%s' %(idx2word(que[i:j], dic2), que[i:j] in sub_idx))
                            if que[i:j] in sub_idx and (not set(que[i:j]) < set(stop_words)):
                                candidate += sub_idx[que[i:j]]
                                if j - i >= sub_j - sub_i:
                                    sub_i, sub_j = i, j
                else:
                    #print('yeah')
                    sub_dis = defaultdict(int)
                    subs = set()
                    for i in range(len(que)):
                        for j in range(i+1, np.min([i+5, len(que)+1])):
                            if not set(que[i:j]) < set(stop_words):
                                #print(idx2word(que[i:j], dic2))
                                que_str = idx2word(que[i:j], dic2)
                                que_ngrams = ngrams(que_str, 3)
                                for que_ngram in que_ngrams:
                                    subs = subs.union(sub_ngram[que_ngram])
                                for sub in subs:
                                    sub_str = idx2word(sub, dic2)
                                    dis = editdistance.eval(sub_str, que_str)
                                    dis = 1. - dis*1./np.max([len(sub_str), len(que_str)])
                                    if sub_dis[sub] < dis:
                                        sub_dis[sub] = dis
                    subs = sorted(sub_dis.items(), key=lambda v: v[1], reverse=True)[:5]
                    # candidate = [(idx2word(sub, dic2), sub) for sub, _ in subs] # change !!!
                    for sub, _ in subs:
                        candidate += sub_idx[sub]
            else:
                for sub in que:
                    candidate += sub_idx[sub]

            for direction, c in candidate:
                s, o, h = squeeze_story(kb[:, :, c], direction, True)
                story += [s]
                obj += [o]
                head += [h]
                # head += [direction] # change !!!
            return (story, obj, (sub_i, sub_j), head)

        new_story, new_obj, new_head = [], [], []
        story, obj, ij, head = search_path(que, sub_idx, kb, hop - 1, is_con)
        candidate = defaultdict(list)
        for line_idx, o in enumerate(obj):
            candidate[line_idx] += sub_idx[o]

        if is_con:
            new_story, new_obj, new_head = story[:], obj[:], head[:] # change !
            for line_idx in candidate:
                for direction, c in candidate[line_idx]:
                    s, o, _ = squeeze_story(kb[:, :, c], direction, False)
                    #if o != que[ij[0]:ij[1]]:
                    new_story += [story[line_idx] + s]
                    new_obj += [o]
                    new_head += [head[line_idx]]

        else:
            story, obj, ij, head = search_path(que, sub_idx, kb, hop - 1, is_con)
            new_story, new_obj, new_head = story[:], obj[:], head[:]
            candidate = defaultdict(list)
            for line_idx, o in enumerate(obj):
                o = tuple([ dictionary[w] for w in idx2word(o, dictionary2).split(' ') ])
                candidate[line_idx] += sub_idx[o]
                for direction, c in candidate[line_idx]:
                    s, o, _ = squeeze_story(kb[:, :, c], direction, True)
                    new_story += [s]
                    new_obj += [o]
                    new_head += [head[line_idx]]

        return (new_story, new_obj, ij, new_head)

    story, story2idx, idx, head = ([], {}, 0, [])
    obj = defaultdict(set)
    raw_story, raw_obj, ij, raw_head = search_path(que, sub_idx, kb, hop, is_con)
    # head = raw_head # change !!!
    for line_idx, s in enumerate(raw_story):
        if s not in story2idx:
            #print(idx2word(s, dic2))
            story += [s]
            head += [raw_head[line_idx]]
            story2idx[s] = idx
            idx += 1
        obj[story2idx[s]].add(raw_obj[line_idx])

    obj = [list(obj[i]) for i in range(len(obj)) ]
    return (story, obj, head, head)

def array2tuple(a):
    b = ()
    for _, k in enumerate(a):
        if k == 0:
            break
        else:
            b += (k,)
    return b

def get_F1(probs, bs, bt, bo, dic2, batch):
    acces = []
    preds = []
    if isinstance(bt[0], tuple):
        for i in range(probs.shape[0]):
            ans = []
            rel = []
            top_index = argmax_all(probs[i, :])
            for j in top_index:
                if j < len(bo[i]):
                    an = bo[i][j] if 1 else []
                    ans += [idx2word(w, dic2) for w in an ]
                    rel += [idx2word(array2tuple(bs[(i, j)]), dic2)]

            y_out = [
             '***'] + list(set(rel)) + ['***'] + [('/').join(list(set(ans)))]
            preds += [y_out]
            if set(ans) == set(bt[batch[i]]):
                acces += [1.0]
            else:
                acces += [0.0]
    else:
        if isinstance(bt[0], str):
            for i in range(probs.shape[0]):
                ans = []
                rel = []
                top_index = argmax_all(probs[i, :])
                # for j in range(len(bo[i])):
                #     print('%s\t%s' %(idx2word(array2tuple(bs[(i, j)]), dic2), probs[i, j]))
                for j in top_index:
                    if j < len(bo[i]):
                        an = bo[i][j] if 1 else []
                        ans += [idx2word(w, dic2) for w in an ]
                        rel += [idx2word(array2tuple(bs[(i, j)]), dic2)]

                y_out = [
                 '***'] + list(set(rel)) + ['***'] + [('/').join(list(set(ans)))]
                preds += [y_out]
                # if bt[batch[i]] in rel[0] and len(rel) == 1:
                #     acces += [1.0]
                # else:
                #     acces += [0.0]
                my_pred = re.sub('parents -1', 'children', y_out[1])
                my_pred = re.sub('children -1', 'parents', my_pred)
                my_pred = re.sub('spouse -1', 'spouse', my_pred)
                my_pred = re.sub('-1', 'inverse', my_pred)
                if bt[batch[i]] == my_pred:
                    acces += [1.0]
                else:
                    acces += [0.0]
                # print(bt[batch[i]])
                # exit()
    #print(y_out)
    return (acces, preds)

def argmax_all(l):
    m = sorted(l)[::-1][:1]
    return [ i for i, j in enumerate(l) if j == m][:1]


def get_filter_F1(target_data, y, alias):
    if target_data[2] in y:
        acc = 1.0
    else:
        acc = 0.0
    return acc


def idx2answer(idxs, dictionary2, order=False):
    if order:
        out = []
        for idx in idxs:
            text = []
            for i in idx:
                text += [dictionary2[i]]

            out += [(' ').join(text)]

    else:
        out = set()
        for idx in idxs:
            text = []
            for i in idx:
                text += [dictionary2[i]]

            out.add((' ').join(text))

    return out


def all_argmax(my_list, start=0, end=1):
    m = sorted(my_list)[::-1][start:end]
    return [ i for i, j in enumerate(my_list) if j in m ]


def save(dirname, session, saver):
    """
    Persist a model's information
    """
    tensorflow_file = os.path.join(dirname, 'model')
    saver.save(session, tensorflow_file)


def print_pred(preds, idx, evals):
    idx = sorted(range(len(idx)), key=lambda x: idx[x])
    pred_text = []
    for i in range(len(idx)):
        text = []
        for j in range(len(preds[idx[i]])):
            w = preds[idx[i]][j]
            text += [w]

        pred_text += [str(i + 1) + '\t' + str(evals[idx[i]]) + '\t' + ('\t').join(text)]

    return pred_text


def extract_top_candidate(candidates, probs, top_index, batch):
    top_candidates = []
    top_probs = []
    for i in range(len(top_index)):
        top_candidate = []
        top_prob = []
        for j in range(len(top_index[i])):
            try:
                top_candidate += candidates[(batch[i], top_index[i][j])]
                top_prob += [probs[i, :][top_index[i][j]]]
            except:
                pass

        top_candidates += [top_candidate]
        top_probs += [top_prob]

    return (top_candidates, top_probs)


# global stop_words ## Warning: Unused global
