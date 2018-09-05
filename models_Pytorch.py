import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from advanced_layers_Pytorch import Z_Layer, Ranker, Ranker2
import time
import numpy as np
import json
from util import *

class Model2(nn.Module):

    def __init__(self, args, emb):
        super(Model2, self).__init__()
        self.args = args
        self.emb = emb
        self.ranker = Ranker2(args, emb, args.learning_rate, args.dropout)
        self.train_step_enc = torch.optim.Adam(self.ranker.parameters(),
                                          lr = self.ranker.lr)

    def forward(self, bx, by, bs, bo, bm, ans, kb, sub_idx, dic, hop = 1, bd = None,
        alias = None, golden_num = None):
        dic, dic2 = dic
        self.ranker.train()
        vbx = Variable(torch.LongTensor(bx), requires_grad=False)
        vbs = Variable(torch.LongTensor(bs), requires_grad=False)
        if bd is not None and isinstance(bd, tuple):
            vdep1 = Variable(torch.LongTensor(bd[0]), requires_grad=False)
            vdep2 = Variable(torch.LongTensor(bd[1]), requires_grad=False)
            vdep = (vdep1, vdep2)
        elif bd is not None:
            vdep = Variable(torch.LongTensor(bd), requires_grad=False)

        self.train_step_enc.zero_grad()

        global grads
        grads = {}
        ytables = []
        t, top, stop, threshold = 0, 5, 0, .5
        inputs = self.ranker.encoder(vbx)
        vbyy = Variable(torch.FloatTensor([0]), requires_grad=False)
        accum_stop = Variable(torch.FloatTensor([0]), requires_grad=True)
        time1 = time.time()
        #hop = np.random.randint(1, 4)
        while t < hop and stop < threshold:#stop < 0.3
            #print('bs size %s' %str(bs.shape))
            if t == 0:
                match_inputs = torch.zeros_like(inputs)
                dep_hc = Variable(torch.zeros((inputs.size()[0], 1, inputs.size()[2], 1)), requires_grad=False)
            if bd is not None:
                vlogits, inputs, vstop, dep_hc = self.ranker.ranker(inputs, vbs,
                    t, dep_hc, vdep)
            else:
                vlogits, inputs, vstop, dep_hc = self.ranker.ranker(inputs, vbs,
                    t, dep_hc)#dep_hc
            stop = vstop.data.numpy()
            # if np.max(by) == 1.:
            #     vbyy[0] = 1.
            # alternative
            if t == (golden_num - 1):
                vbyy[0] = 1
            #print(vstop)
            temp_stop = -(1.*vbyy*torch.log(torch.clamp(vstop, min = 1e-10))+
                (1.-vbyy)*torch.log(torch.clamp(1-vstop, min = 1e-10)))
            accum_stop = accum_stop + temp_stop
            print('target stop %s\tstop %s\tloss%s' %(vbyy.data.numpy(), stop, temp_stop.data.numpy()))
            if t > 0:
                vbss = Variable(torch.LongTensor(bs), requires_grad=False)
                vlogits = self.ranker.final_probs(vlogits, vprev_logits, vbss)
            else:
                vlogits = torch.squeeze(vlogits, 1)
            if t+1 < hop and stop < threshold:
                logits = vlogits.data.numpy()
                #print('%s\t%s\t%s' %(t, str(logits.shape), str(bs.shape)))
                by, nbs, bo, bm, ytable, top_logits, bs = obtain_next_xyz(logits,
                    bx, bs, bo, bm, ans, kb, sub_idx, [dic, dic2],
                    is_prev_log = True, top=top, alias = alias)
                vbs = Variable(torch.LongTensor(nbs), requires_grad=False)
                vlogits_mask = Variable(torch.ByteTensor(top_logits), requires_grad=False)
                vprev_logits = torch.torch.masked_select(vlogits, vlogits_mask).\
                                    view(-1, np.min([top_logits.shape[1], top]))
                B, _, QL, n_d = inputs.size()
                vlogits_mask = torch.unsqueeze(torch.unsqueeze(vlogits_mask, -1), -1)
                #print('mask %s' %str(match_inputs.size()))
                inputs = torch.torch.masked_select(inputs, vlogits_mask).\
                            view(B, np.min([top_logits.shape[1], top]), QL, -1)
                dep_hc = torch.torch.masked_select(dep_hc, vlogits_mask).\
                            view(B, np.min([top_logits.shape[1], top]), QL, 1) #coverage !!!

            t += 1

        by /= np.expand_dims(np.maximum(np.sum(by, 1), 1.e-10), 1)
        vby = Variable(torch.FloatTensor(by), requires_grad=False)
        cost = self.ranker.obtain_reward(vby, vlogits, accum_stop)
        print('model forward %s' %(time.time() - time1))

        time1 = time.time()
        cost.backward(retain_graph = False)
        self.train_step_enc.step()
        cost = self.ranker.loss.data.numpy()
        print('model backward %s' %(time.time() - time1))
        #emb = self.ranker.emb_init.weight.data.numpy()
        #emb = emb / np.expand_dims(np.linalg.norm(emb, axis = 1), axis = 1)
        #print(np.sum(emb[10, :]**2))
        #self.ranker.emb_init.weight.data.copy_(torch.from_numpy(emb))

        preds = vlogits.data.numpy()
        loss = cost
        display = self.ranker.x_masks.data.numpy()
        return cost, loss, cost, preds, t, bs, bo, by

    def check_accuracy(self, bx, by, bs, bo, bm, ans, kb, sub_idx, dic, hop = 1,
        bd = None, alias = None):
        time1 = time.time()
        dic, dic2 = dic
        self.ranker.eval()
        vbx = Variable(torch.LongTensor(bx), requires_grad=False)
        vbs = Variable(torch.LongTensor(bs), requires_grad=False)
        if bd is not None and isinstance(bd, tuple):
            vdep1 = Variable(torch.LongTensor(bd[0]), requires_grad=False)
            vdep2 = Variable(torch.LongTensor(bd[1]), requires_grad=False)
            vdep = (vdep1, vdep2)
        elif bd is not None:
            vdep = Variable(torch.LongTensor(bd), requires_grad=False)

        bstables = []
        botables = []
        bytables = []
        bmtables = []
        predtables = []
        t, top, stop, threshold = 0, 5, 0, .5
        inputs = self.ranker.encoder(vbx)
        #print('prepare %s' %(time.time() - time1))
        #hop = np.random.randint(1, 4)
        while t < hop and stop < threshold:
            time1 = time.time()
            bstables += [bs]
            botables += [bo]
            bytables += [by]
            bmtables += [bm]
            if t == 0:
                match_inputs = torch.unsqueeze(torch.zeros_like(inputs), 1)
                dep_hc = Variable(torch.zeros((inputs.size()[0], 1, inputs.size()[2], 1)), requires_grad=False)
            if bd is not None:
                vlogits, inputs, vstop, dep_hc = self.ranker.ranker(inputs, vbs,
                    t, dep_hc, vdep)
            else:
                vlogits, inputs, vstop, dep_hc = self.ranker.ranker(inputs, vbs,
                    t, dep_hc)
            stop = vstop.data.numpy()
            # print('stop %s' %(stop))
            if t > 0:
                vbss = Variable(torch.LongTensor(bs), requires_grad=False)
                vlogits = self.ranker.final_probs(vlogits, vprev_logits, vbss)
            else:
                vlogits = torch.squeeze(vlogits, 1)
            #print('ranker %s' %(time.time() - time1))
            time1 = time.time()
            if t+1 < hop and stop < threshold:
                logits = vlogits.data.numpy()
                by, nbs, bo, bm, ytable, top_logits, bs = obtain_next_xyz(logits,
                    bx, bs, bo, bm, ans, kb, sub_idx, [dic, dic2],
                    is_prev_log = True, top=top, alias = alias)
                predtables += [logits]
                vbs = Variable(torch.LongTensor(nbs), requires_grad=False)
                vlogits_mask = Variable(torch.ByteTensor(top_logits), requires_grad=False)
                vprev_logits = torch.torch.masked_select(vlogits, vlogits_mask).\
                                    view(-1, np.min([top_logits.shape[1], top]))
                B, _, QL, n_d = inputs.size()
                vlogits_mask = torch.unsqueeze(torch.unsqueeze(vlogits_mask, -1), -1)
                #print('mask %s' %str(match_inputs.size()))
                inputs = torch.torch.masked_select(inputs, vlogits_mask).\
                            view(B, np.min([top_logits.shape[1], top]), QL, -1)
                dep_hc = torch.torch.masked_select(dep_hc, vlogits_mask).\
                            view(B, np.min([top_logits.shape[1], top]), QL, 1)
            t += 1

        preds = vlogits.data.numpy()
        predtables += [preds]
        return predtables, bstables, botables, bmtables, bytables,

class Train_loop(object):

    def __init__(self, args, emb):
        self.args = args
        self.emb = emb
        self.model = Model2(args = args, emb = emb)
        self.save_para = self.model.state_dict().keys()

    def train(self, train, dev, test, kb, sub_idx, dic, outfile, rel = None,
        deps = None, alias = None):
        args = self.args
        dropout = args.dropout
        padding_id = 0
        dic2 = {}
        for d in dic:
            dic2[dic[d]] = d
        # print([idx2word(w, dic2) for w in sub_idx.keys()[:100]])

        min_val_err, min_test_err = -1, -1
        hop = 3#int(args.hop)

        if args.load_id:
            try:
                self.model.load_state_dict(torch.load(
                        '%s/model%s.ckpt' %(outfile, args.load_id)), strict = False)
                print('successfully load pre-trained parameters ...')
            except:
                print('fail to load pre-trained parameters ...')
        if 'audio' in args.train_q:
            sub_ngram = read_topic('./data/mix-hop/audio', dic)
               # sub_ngram = subidx2ngrams(sub_idx, dic, dic2, n=3)
        else:
            sub_ngram = [None, None, None]

        if args.is_train:
            train_shuffle_batch = create_batches(len(train[0]), args.batch)#[:5]
            train_shuffle_batch = train_shuffle_batch*args.max_epochs
            # train_shuffle_batch = create_batches2('%s/pred_train_bolen.txt'
            #     %(outfile), 10000, args.batch)
        N = len(train_shuffle_batch) if args.is_train else 0

        max_epochs = int(N/500)
        for ep in range(max_epochs):#args.max_epochs

            processed = 0
            train_cost = 0.
            train_loss = 0.
            train_zdiff = 0.
            train_err = 0.
            train_preds = []
            train_eval = []
            train_bo_len = []

            start = ep*500
            end = (ep + 1)*500
            for i in range(start, end): # 3478, 4099
                dep = deps[0] if deps else None
                bx, by, bs, bo, bm, be, ba, bd = obtain_xys(train, kb, sub_idx,
                    train_shuffle_batch[i], [dic, dic2], dep = dep, alias = alias,
                    sub_ngram = sub_ngram[0])
                #print(bs)
                print('>>> batch: %s\ttask %s\tsave id %s\tbx %s\tby %s\tbs %s'
                    %(i, args.task, args.save_id, str(bx.shape), str(by.shape), str(bs.shape)))

                golden = rel[0] if args.eval == "rel_acc" else train[1]
                cost, loss, zdiff, probs, display, bs, bo, by = self.model.forward(bx,
                        by, bs, bo, bm, ba, kb, sub_idx, [dic, dic2],
                        hop =hop, bd = bd, alias = alias, golden_num = golden[0][train_shuffle_batch[i][0]])

                err, pred = get_F1(probs, bs, golden[1], bo, dic2, train_shuffle_batch[i],
                    alias = alias)
                print('##############')
                print(pred[0][1])
                print(golden[1][train_shuffle_batch[i][0]])
                # if i == 30:
                #     exit()

                k = len(by)
                processed += k
                train_cost += cost
                train_loss += loss
                train_zdiff += np.sum(zdiff)
                train_err += np.sum(err)
                train_preds += pred
                train_eval += err

                for b in range(k):
                    print('index %s\tgenerate num %s\teval %s\ttarget %s\t%s-%s'
                        %(train_shuffle_batch[i][b], len(bo[b]), err[b], np.sum(by[b]),
                        display, golden[0][train_shuffle_batch[i][b]]))
                    train_bo_len += [len(bo[b])]
                    #split2hop[len(golden[train_shuffle_batch[i][b]].split(' '))]
                    #print(pred)
                if (i+1) % 10 == 0:
                    embedding = self.model.ranker.emb_init.weight.data.numpy()
                    np.save('%s/weights%s' %(outfile, args.save_id), embedding)
                    saver = get_weights_and_biases(self.model, self.save_para)
                    torch.save(saver, '%s/model%s.ckpt' %(outfile, args.save_id))
                #stop

            train_cost /= np.max([processed, 1.e-10])
            train_loss /= np.max([processed, 1.e-10])
            train_err /= np.max([processed, 1.e-10])
            train_zdiff /= np.max([processed, 1.e-10])
            # train_preds = print_pred(train_preds, shuffle_batch, train_eval)
            #train_preds = print_pred(train_bo_len, train_shuffle_batch)
            #save_pred('%s/pred%s_train_bolen.txt' %(outfile, args.save_id), train_preds)

            message = '%d | train loss %g(%g)(%g) | train eval: %g ' \
                    %(ep+1, train_cost, train_loss, train_zdiff, train_err)

            print('Evaluation ... ')

            if args.only_eval:
                shuffle_batch = [[6]]
            else:
                shuffle_batch = create_batches(len(dev[0]), args.batch)#[:500]
                # shuffle_batch = create_batches2('%s/pred_dev_bolen.txt'
                #     %(outfile), 10000, args.batch)

            N = len(shuffle_batch)
            valid_err = 0.
            processed = 0
            valid_eval = []
            valid_preds = []
            valid_bo_len = []
            for i in range(0): #687
                dep = deps[1] if deps else None
                bx, by, bs, bo, bm, be, ba, bd = obtain_xys(dev, kb, sub_idx,
                    shuffle_batch[i], [dic, dic2], dep = dep, alias = alias,
                    is_train=False, sub_ngram = sub_ngram[1])
                print('>>> batch: %s\tbx %s\tby %s\tbs %s' %(i, bx.shape,
                    by.shape, bs.shape))

                probs, bs, bo, bm, by = self.model.check_accuracy(bx,
                        by, bs, bo, bm, ba, kb, sub_idx, [dic, dic2],
                        hop =hop, bd = bd, alias = alias)

                golden = rel[1] if args.eval == "rel_acc" else dev[1]
                err, pred = get_F1(probs[-1], bs[-1], golden[1], bo[-1], dic2,
                    shuffle_batch[i], alias = alias)

                k = len(by[-1])
                processed += k
                valid_err += np.sum(err)
                valid_eval += err
                valid_preds += pred

                #stop
                for b in range(k):
                    # print('\n>> Q: %s\t>>> A:%s' %(idx2word(bx[b, :], dic2),
                    #     dev[1][shuffle_batch[i][b]]))
                    # print(golden[shuffle_batch[i][b]])
                    # print(pred)
                    if args.only_eval:
                        print('\n>> Q: %s\t>>> A:%s' %(idx2word(bx[b, :], dic2),
                            dev[1][shuffle_batch[i][b]]))
                        #print(bx[b, :])
                        #print(be[b, :])
                        for t in range(len(by)):
                            for j in range(bs[t].shape[1]):
                                if bs[t][b, j, 0] != 0:
                                    try:
                                        prob = probs[t][b, j]
                                    except:
                                        prob = ''
                                    print('hop %s\t%s\t%s\t%s' %(t,
                                        idx2word(bs[t][b, j, :], dic2),
                                        by[t][b, j], prob))
                                    if j < len(bo[t][b]):
                                        print('--->%s'%str([idx2word(w, dic2) for
                                            w in bo[t][b][j][:10]]))
                                        # print(bm[t][b][j])
                                    #print(display[b, j, :10, :10])
                    print('batch %s\tindex %s\tgenerate num %s\teval %s\ttarget %s'
                        %(i, shuffle_batch[i][b], len(bo[-1][b]), err[b], np.sum(by[-1][b])))
                    valid_bo_len += [len(bo[-1][b])]

            valid_err /= np.max([processed, 1.e-10])
            message += ' | val eval: %g ' %valid_err
            print(message)

            if not args.only_eval:
                pass#valid_preds = print_pred(valid_preds, shuffle_batch, valid_eval)
                #save_pred('%s/pred%s_dev.txt' %(outfile, args.save_id), valid_preds)
                # valid_preds = print_pred(valid_bo_len, shuffle_batch)
                # save_pred('%s/pred%s_dev_bolen.txt' %(outfile, args.save_id), valid_preds)

            test_err = 0.
            if valid_err > -.5:
                shuffle_batch = create_batches(len(test[0]), args.batch)[:3000]
                # shuffle_batch = create_batches2('%s/pred_test_bolen.txt'
                #     %(outfile), 10000, args.batch)

                processed = 0
                test_eval = []
                test_preds = []
                test_bo_len = []
                N = len(shuffle_batch)
                for i in range(N):
                    print(shuffle_batch[i])
                    dep = deps[2] if deps else None
                    bx, by, bs, bo, bm, be, ba, bd = obtain_xys(test, kb, sub_idx,
                        shuffle_batch[i], [dic, dic2], dep = dep, alias = alias,
                        is_train=False, sub_ngram = sub_ngram[2])
                    # print('>>> batch: %s\tbx %s\tby %s\tbs %s' %(i, bx.shape,
                    #     by.shape, bs.shape))

                    probs, bs, bo, bm, by = self.model.check_accuracy(bx,
                        by, bs, bo, bm, ba, kb, sub_idx, [dic, dic2],
                        hop =hop, bd = bd, alias = alias)

                    golden = rel[2] if args.eval == "rel_acc" else test[1]
                    err, pred = get_F1(probs[-1], bs[-1], golden[1], bo[-1], dic2,
                        shuffle_batch[i], alias = alias)

                    k = len(by[-1])
                    processed += k
                    test_err += np.sum(err)
                    test_eval += err
                    test_preds += pred

                    for b in range(k):
                        print('batch %s\tindex %s\tgenerate num %s\teval %s\ttarget %s'
                            %(i, shuffle_batch[i][b], len(bo[-1][b]), err[b],
                                np.sum(by[-1][b])))
                        test_bo_len += [len(bo[-1][b])]
                    #exit()

                test_err /= np.max([processed, 1.e-10])
                message += ' | test eval: %g ' %test_err

                if test_err > min_test_err:
                    min_test_err = test_err
                    test_preds = print_pred(test_preds, shuffle_batch, test_eval)
                    save_pred('%s/pred%s_test.txt' %(outfile, args.save_id), test_preds)
                    #test_preds = print_pred(test_bo_len, shuffle_batch)
                    #save_pred('%s/pred%s_test_bolen.txt' %(outfile, args.save_id), test_preds)

            print(message)
            if args.is_train:
                if (test_err == 0 and valid_err > min_val_err) or (test_err > 0 and test_err == min_test_err):
                    min_val_err = valid_err
                    embedding = self.model.ranker.emb_init.weight.data.numpy()
                    np.save('%s/weights%s' %(outfile, args.save_id), embedding)
                    saver = get_weights_and_biases(self.model, self.save_para)
                    torch.save(saver, '%s/model%s.ckpt' %(outfile, args.save_id))
                    message += ' (saved model)'
            else:
                stop

            log = open('%s/result%s.txt' %(outfile, args.save_id), 'a')
            log.write(message + "\n")
            log.close()

def get_weights_and_biases(model, save_para):
    state_dict = {}
    old_dict = model.state_dict()
    for var in old_dict:
        if 'emb_init' not in var and var in save_para:
            state_dict[var] = old_dict[var]
    return state_dict

def save_grad(name):
    print('***********')
    def hook(grad):
        grads[name] = grad
    return hook
