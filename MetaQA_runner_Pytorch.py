from options import load_arguments
from util import *
import numpy as np
from time import gmtime, strftime
import os
import pickle
import re

from models_Pytorch import Train_loop

def main(args, load_save = None):
    print("Train and test for task  ..." )
    if load_save:
        outfile = load_save
        args = pickle.load(open('%s/config.pkl' %outfile, 'rb'))
        dic = pickle.load(open('%s/dic.pkl' %outfile, 'rb'))
    else:
        outfile = 'trained_model/' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        dic = {"<unk>": 0, "-1": 1}

    kb_file = './data/mix-hop/kb.txt'
    args.load_id = '_none'
    args.save_id = '_Acc_beamsearch2'
    args.embedding = '/home/yunshi/Word2vec/glove.840B.300d.zip'
    args.eval = 'rel_acc'
    args.with_dep = None
    args.coherent = 0.
    args.max_epochs = 20
    args.zin = 0.
    args.batch = 1
    args.task = load_save
    rel, dep = None, None
    args.learning_rate = 0.0001
    args.hidden_dimension = 200
    args.dropout = 0.3

    if load_save:
        is_WEBQA = True if re.findall('[^/]+(?=/qa)', args.train_q)[0] == 'WBQ' else False
        train_x, train_y, train_e = read_annotations(args.train_q, dic, is_WEBQA)
        dev_x, dev_y, dev_e = read_annotations(args.dev_q, dic, is_WEBQA)
        test_x, test_y, test_e = read_annotations(args.test_q, dic, is_WEBQA)
        if args.with_dep == 'both':
            train_d1 = read_dep(re.sub('.txt', '_path.txt', args.train_q), 'path', dic)
            dev_d1 = read_dep(re.sub('.txt', '_path.txt', args.dev_q), 'path', dic)
            test_d1 = read_dep(re.sub('.txt', '_path.txt', args.test_q), 'path', dic)
            train_d2 = read_dep(re.sub('.txt', '_tag.txt', args.train_q), 'tag', dic)
            dev_d2 = read_dep(re.sub('.txt', '_tag.txt', args.dev_q), 'tag', dic)
            test_d2 = read_dep(re.sub('.txt', '_tag.txt', args.test_q), 'tag', dic)
            print('train_d %s dev_dev %s test_dev %s' %(len(train_d1), len(dev_d1), len(test_d1)))
            dep = ((train_d1, train_d2), (dev_d1, dev_d2), (test_d1, test_d2))
        elif args.with_dep is not None:
            train_d = read_dep(re.sub('.txt', '_%s.txt'%args.with_dep, args.train_q), args.with_dep, dic)
            dev_d = read_dep(re.sub('.txt', '_%s.txt'%args.with_dep, args.dev_q), args.with_dep, dic)
            test_d = read_dep(re.sub('.txt', '_%s.txt'%args.with_dep, args.test_q), args.with_dep, dic)
            print('train_d %s dev_dev %s test_dev %s' %(len(train_d), len(dev_d), len(test_d)))
            dep = (train_d, dev_d, test_d)
        print('train %s dev %s test %s' %(len(train_x), len(dev_x), len(test_x)))
        # train_x, train_y, train_e, dev_x, dev_y, dev_e, test_x, test_y, test_e = \
        #    pickle.load(open('%s/q.pkl' %outfile, 'r'))
        args.is_train = 1
        args.only_eval = 0

        if is_WEBQA:
            kb, sub_idx  = load_kbsubidx('%s/kb.gz' %outfile)
        else:
            kb, sub_idx = pickle.load(open('%s/kb.pkl' %outfile, 'rb'))
        if os.path.isfile('%s/weights%s.npy' %(outfile, args.load_id)):
            emb = np.array(np.load('%s/weights%s.npy' %(outfile, args.load_id)))
        else:
            emb = initialize_vocab(dic, args.embedding)
            np.save("%s/weights%s" %(outfile, args.save_id), emb)
    else:
        args.train_q = 'data/mix-hop/vanilla/qa_train.txt'
        args.dev_q = 'data/mix-hop/vanilla/qa_dev.txt'
        args.test_q = 'data/mix-hop/vanilla/qa_test.txt'
        is_WEBQA = True if re.findall('[^/]+(?=/qa)', args.train_q)[0] == 'WBQ' else False
        train_x, train_y, train_e = read_annotations(args.train_q, dic, is_WEBQA)
        dev_x, dev_y, dev_e = read_annotations(args.dev_q, dic, is_WEBQA)
        test_x, test_y, test_e = read_annotations(args.test_q, dic, is_WEBQA)
        kb, sub_idx = read_kb(kb_file, dic, is_WEBQA)
        emb = initialize_vocab(dic, args.embedding)

        pickle.dump([train_x, train_y, train_e, dev_x, dev_y, dev_e,
            test_x, test_y, test_e],
            open('%s/q.pkl' %outfile, 'wb'), protocol = 2)
        if is_WEBQA:
            save_kbsubidx('%s/kb.gz' %outfile, kb, sub_idx)
        else:
            pickle.dump([kb, sub_idx],
              open('%s/kb.pkl' %outfile, 'wb'), protocol = 2)

        pickle.dump(dic, open('%s/dic.pkl' %outfile, 'wb'), protocol = 2)
        np.save("%s/weights%s" %(outfile, args.save_id), emb)

    print('train: %s dev: %s test: %s' %(len(train_x), len(dev_x), len(test_x)))
    args.hop = re.findall('[^/]+(?=-hop)', args.train_q)[0]
    pickle.dump(args, open('%s/config.pkl' %outfile, 'wb'), protocol = 2)
    save_config(args, '%s/config.txt' %outfile)
    #stop
    alias = read_alias('./data/WebQA_alias.txt') if is_WEBQA else None

    if args.eval == 'rel_acc':
        train_r = read_golden_rel('./data/%s-hop/qa_train_qtype.txt' %args.hop, True)
        dev_r = read_golden_rel('./data/%s-hop/qa_dev_qtype.txt' %args.hop, True)
        test_r = read_golden_rel('./data/%s-hop/qa_test_qtype.txt' %args.hop, True)
        #print('train %s dev %s test %s' %(len(train_r[0]), len(dev_r[0]), len(test_r[0])))
        rel = (train_r, dev_r, test_r)
        #print(test_r[1][4263])

    print('Parser Arguments')
    for key, value in args.__dict__.items():
        print(u'{0}: {1}'.format(key, value))

    train_loop = Train_loop(args = args, emb = emb)

    train_loop.train((train_x, train_y, train_e),
                    (dev_x, dev_y, dev_e),
                    (test_x, test_y, test_e),
                    kb, sub_idx, dic, outfile, rel = rel, deps = dep, alias = alias)

if __name__ == '__main__':
    args = load_arguments()
    # reset zie graph

    if args.task == 1: # 1-hop vanilla
        main(args, load_save = 'trained_model/2018-02-15-09-07-04')
    elif args.task == 2: # 1-hop ntm
        main(args, load_save = 'trained_model/2018-02-08-03-46-16')
    elif args.task == 3: # 2-hop vanilla
        main(args, load_save = 'trained_model/2018-02-13-08-57-48')
    elif args.task == 4: # 2-hop ntm
        main(args, load_save = 'trained_model/2018-02-07-05-35-08')
    elif args.task == 5: # 3-hop vanilla
        main(args, load_save = 'trained_model/2018-02-07-05-21-07')
    elif args.task == 6: # 3-hop ntm
        main(args, load_save = 'trained_model/2018-02-13-08-54-28')
    elif args.task == 7: # x-hop vanilla
        main(args, load_save = 'trained_model/2018-03-22-03-39-56')
    elif args.task == 8: # x-hop ntm
        main(args, load_save = 'trained_model/2018-03-18-09-16-06')
    elif args.task == 9: # x-hop audio
        main(args, load_save = 'trained_model/2018-04-10-02-55-47')
    elif args.task == 10: # x-hop path questions
        main(args, load_save = 'trained_model/2018-07-30-10-49-19')
    elif args.task == 11: # x-hop path questions large
        main(args, load_save = 'trained_model/2018-07-31-09-27-20')
    elif args.task == 12: # x-hop PathQuestions
        main(args, load_save = 'trained_model/2018-07-31-10-04-33')
    elif args.task == 0:
        main(args)
