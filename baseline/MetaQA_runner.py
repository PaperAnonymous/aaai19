import glob
import os
import random
import sys

import argparse
import numpy as np
import pickle
from time import gmtime, strftime
from collections import defaultdict

from config import Config
from train_test_MetaQA import train
from util import *

seed_val = 42
np.random.seed(seed_val)  # for reproducing

def run_task(data_dir, arg, load_save):
    """
    Train and test for each task
    """
    print("Train and test for task  ..." )
    if load_save:
        outfile = load_save
        general_config = pickle.load(open('%s/config.pkl' %outfile, 'rb'))
        dictionary = pickle.load(open('%s/dic.pkl' %outfile, 'rb'))
        general_config.nepochs = 80
        general_config.train_config['max_grad_norm'] = 5
        general_config.train_config['in_dim'] = 20
        general_config.train_config['task'] = arg[0]
        general_config.train_config['add_sib'] = arg[5]
        general_config.train_config['eval'] = 'rel_acc'
        general_config.train_config['model'] = 'matching'
        general_config.train_config['max_hop'] = 2
    else:
        outfile = 'trained_model/' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        if not os.path.exists(outfile):
            os.makedirs(outfile)
        dictionary = {"nil": 0, "-1": 1}

    # Parse data
    train_files_sq = glob.glob('%s/mix-hop/WC2014/qa_train.txt' % (data_dir))
    valid_files_sq = glob.glob('%s/mix-hop/WC2014/qa_dev.txt' % (data_dir))
    test_files_sq  = glob.glob('%s/mix-hop/WC2014/qa_test.txt' % (data_dir))
    story_files = glob.glob('%s/mix-hop/WC2014/kb.txt' %(data_dir))

    sib_idx, sib_dic = None, None
    print("""parameter setting:
        task %s, in_dim %s, dropout %s, whether_load %s, only_eval %s,
        add_sib %s, load_id %s, is_train %s, post_processing %s, mode %s, save_id %s"""
        %tuple(arg))

    #parase = parse_paraphrases(para_files, dictionary)

    if load_save:
        train_questions, train_qstory = parse_QA_task(train_files_sq, dictionary, False)
        valid_questions, valid_qstory = parse_QA_task(valid_files_sq, dictionary, False)
        test_questions, test_qstory = parse_QA_task(test_files_sq, dictionary, False)
        pickle.dump([train_questions, valid_questions, test_questions, train_qstory, \
          valid_qstory, test_qstory], open('%s/question.pkl' %outfile, 'wb'), protocol = 2)
        general_config.train_config['max_hop'] = 2
        # train_questions, valid_questions, test_questions, train_qstory, valid_qstory, \
        #    test_qstory = pickle.load(open('%s/question.pkl' %outfile, 'rb'))
        general_config.train_config['max_words'] = np.max([
            len(x) for x in train_questions])
        if general_config.train_config['model'] == 'memn2n':
            story, so_idx = parse_freebase_memn2n(story_files, dictionary)
            general_config.train_config['voc_sz'] = len(dictionary)
        else:
            story, so_idx = pickle.load(open('%s/kb.pkl' %outfile, 'rb'))
        #print(story[0, :, :100])
        model, loss = build_model_tf(general_config, dictionary, arg[6], arg[-1], outfile)
    else:
        train_questions, train_qstory = parse_QA_task(train_files_sq, dictionary)
        valid_questions, valid_qstory = parse_QA_task(valid_files_sq, dictionary)
        test_questions, test_qstory = parse_QA_task(test_files_sq, dictionary)
        pickle.dump([train_questions, valid_questions, test_questions, train_qstory, \
          valid_qstory, test_qstory], open('%s/question.pkl' %outfile, 'wb'), protocol = 2)
        story, so_idx = parse_freebase(story_files, dictionary)
        pickle.dump([story, so_idx], open('%s/kb.pkl' %outfile, 'wb'), protocol = 2)
        pickle.dump(dictionary, open('%s/dic.pkl' %outfile, 'wb'), protocol= 2)
        general_config = Config(story, train_questions, train_qstory, dictionary, outfile)
        pickle.dump(general_config, open('%s/config.pkl' %outfile, 'wb'), protocol= 2)
        model, loss = build_model_tf(general_config, dictionary, arg[6], arg[-1], outfile)

    if general_config.train_config['eval'] == 'rel_acc':
        train_r = read_golden_rel('./data/mix-hop/WC2014/qa_train_qtype.txt')
        dev_r = read_golden_rel('./data/mix-hop/WC2014/qa_dev_qtype.txt')
        test_r = read_golden_rel('./data/mix-hop/WC2014/qa_test_qtype.txt')
        rel = (train_r, dev_r, test_r)
    else:
        rel = None

    train(story, so_idx, [train_questions, valid_questions, test_questions], \
        [train_qstory, valid_qstory, test_qstory], \
        model, general_config, dictionary, arg, rel = rel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", default="""./data""",
                        help="path to dataset directory (default: %(default)s)")
    parser.add_argument("-dim", "--in_dim", default="150", type=int,
                       help="hidden dimension of neural network (default: %(default)s)")
    parser.add_argument("-dp", "--dropout", default="0.8", type=float,
                       help="dropout ratio of neural network (default: %(default)s)")
    parser.add_argument("-t", "--task", default="1", type=int,
                       help="default task (default: %(default)s)")
    parser.add_argument("-l", "--load", default="1", type=int,
                       help="whether load datasets (default: %(default)s)")
    parser.add_argument("-o", "--only_eval", default="0", type=int,
                       help="whether just for evaluation (default: %(default)s)")
    parser.add_argument("-i", "--is_train", default="1", type=int,
                       help="whether train models (default: %(default)s)")
    parser.add_argument("-p", "--is_post_processing", default="1", type=int,
                       help="whether do post-processing (default: %(default)s)")
    parser.add_argument("-jl", "--joint_learn", default="-2", type=int,
                       help="whether add KB info (default: %(default)s)")
    parser.add_argument("-id", "--tid", default="simple150_copy", type=str,
                       help="task id (default: %(default)s)")
    parser.add_argument("-s", "--add_sib", default="0", type=int,
                       help="whether add sib info (default: %(default)s)")
    parser.add_argument("-li", "--load_id", default="simple150", type=str,
                       help="which model should be loaded (default: %(default)s)")
    args = parser.parse_args()

     #Check if data is available
    data_dir = args.data_dir
    arg = [args.task, args.in_dim, args.dropout, args.load, args.only_eval, args.add_sib, \
        args.load_id, args.is_train, args.is_post_processing, args.joint_learn, args.tid]

#def main(data_dir, arg):
    if not os.path.exists(data_dir):
        print("The data directory '%s' does not exist. Please download it first." % data_dir)
        sys.exit(1)

    print("Using data from %s" % data_dir)
    if arg[0] == 1:#1-hop vannila
        run_task(data_dir, arg, load_save = 'trained_model/2018-01-18-10-39-27')
    elif arg[0] == 2:#1-hop ntm
        run_task(data_dir, arg, load_save = 'trained_model/2018-01-18-03-37-29')
    elif arg[0] == 3: # 2-hop vanilla
        run_task(data_dir, arg, load_save = 'trained_model/2018-03-03-11-28-52')
    elif arg[0] == 4:#2-hop ntm
        run_task(data_dir, arg, load_save = 'trained_model/2018-01-18-13-20-39')
    elif arg[0] == 5: # 3-hop vanilla
        run_task(data_dir, arg, load_save = 'trained_model/2018-03-03-16-03-00')
    elif arg[0] == 6:#3-hop ntm
        run_task(data_dir, arg, load_save = 'trained_model/2018-01-18-13-25-14')
    elif arg[0] == 7: # vanilla
        run_task(data_dir, arg, load_save = 'trained_model/2018-03-22-05-12-43')
    elif arg[0] == 8: # ntm
        run_task(data_dir, arg, load_save = 'trained_model/2018-03-21-03-23-43')
    elif arg[0] == 9: #
        run_task(data_dir, arg, load_save = 'trained_model/2018-04-11-12-32-25')
    elif arg[0] == 10: # P-athQuestions
        run_task(data_dir, arg, load_save = 'trained_model/2018-08-13-10-13-48')
    # elif arg[0] == 11:
    #     run_task(data_dir, arg, load_save = 'trained_model/2018-08-03-01-53-56')
    elif arg[0] == 12: # W-C2014
        run_task(data_dir, arg, load_save = 'trained_model/2018-08-13-10-16-17')
    elif arg[0] == 0:
        run_task(data_dir, arg, load_save = None)
