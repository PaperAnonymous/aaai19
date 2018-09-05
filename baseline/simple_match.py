import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
tf.set_random_seed(1234)

def BiRNN(sequence, num_hidden, sequence_w_len = None, reuse = None, keep_prob = 0.8, scope = None):
    cell_fw = DropoutWrapper(tf.contrib.rnn.LSTMCell(num_hidden, reuse=reuse),
                             output_keep_prob=keep_prob,
                             dtype=tf.float32)
    cell_bw = DropoutWrapper(tf.contrib.rnn.LSTMCell(num_hidden, reuse=reuse),
                             output_keep_prob=keep_prob,
                             dtype=tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          sequence,
                                                          sequence_length=sequence_w_len,
                                                          dtype=tf.float32,
                                                          scope = scope)
    return tf.concat(outputs, 2)

def RNN(sequence, num_hidden, sequence_w_len = None, reuse = None, keep_prob = 0.8, scope = None):
    cell = DropoutWrapper(tf.contrib.rnn.LSTMCell(num_hidden, reuse=reuse),
                          output_keep_prob=keep_prob,
                          dtype=tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, sequence,
                                    sequence_length=sequence_w_len,
                                    dtype=tf.float32,
                                    scope = scope)

    return outputs

def CNN(sequence, num_hidden, reuse = None, scope = None):
    # Convolution Layer
    filter_shape = [filter_size, 100, 1, num_hidden]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), \
        name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
    conv = tf.nn.conv2d(x,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name=scope)
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    # Max-pooling over the outputs
    pooled_outputs = tf.nn.max_pool(h,
                                    ksize=[1, 14 - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name=scope)

    # Combine all the pooled features
    h_pool_flat = tf.reshape(pooled_outputs, [-1, num_hidden])
    return h_pool_flat

def foward_ReLU(sequence, num_hidden, reuse = None, keep_prob = 1., scope = None):
    sequence = tf.nn.dropout(sequence, keep_prob)
    output = tf.contrib.layers.fully_connected(sequence, num_hidden, \
                                                tf.nn.relu, reuse = reuse, scope=scope)
    return output

def kl_divergence(p, q):
    return tf.reduce_sum(p * tf.log(tf.maximum(1.e-10, p/q)))

def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)

class Simple_Match_Model(object):

    def __init__(self, general_config, vocab):
        # if load:
        #     session=tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        #     saver = tf.train.Saver(get_weights_and_biases())
        #     saver.restore(session, '%s/model2.ckpt' %outfile)
        #     init_op = tf.initialize_variables([self.A_1])
        #     session.run(init_op, {self.A_1: vocab})
        # else:
        tf.reset_default_graph()
        session=tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
        self._vocab_size = general_config.train_config['voc_sz']
        self._embedding_size = general_config.train_config['in_dim']
        self._max_grad_norm = general_config.train_config['max_grad_norm']
        self._init = tf.random_normal_initializer(stddev=0.1)
        self._name = 'MemN2N'
        self._config = general_config.train_config
        self._reg = None#tf.contrib.layers.l2_regularizer(scale=0.1)

        with tf.device('/gpu:0'):
            self._build_inputs()
            tmp = self._build_vars(vocab)

        self._opt = tf.train.AdagradOptimizer(learning_rate=self._lr)

        # cross entropy
        logits, show, show1 = self._inference_rel_attention(self._stories, self._sibs, \
            self._queries)

        #loss1 = kl_divergence(self._answers, logits)
        #loss1 = kl(logits, self._answers)
        #loss1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self._answers))
        loss1 = tf.reduce_sum((self._answers - logits)**2)
        cross_entropy = loss1
        # loss op
        loss_op = cross_entropy#/tf.log(tf.cast(tf.shape(self._stories)[1], tf.float32))

        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(g, v) if g is None else (tf.clip_by_norm(g, self._max_grad_norm), v) \
            for g,v in grads_and_vars]
        grads_and_vars = [(g, v) if g is None else (add_gradient_noise(g), v) \
            for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                #nil_grads_and_vars.append((zero_nil_slot(g), v))
                pass
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")
        self.A_1 = tf.nn.l2_normalize(self.A_1, 1)

        # predict ops
        #predict_op = tf.argmax(logits, 1, name="predict_op")
        self.predict_op = logits
        self.show = show
        self.show1 = show1

        # assign ops
        self.loss_op = loss_op
        self.train_op = train_op

        self._sess = session
        self._sess.run(tf.global_variables_initializer(), feed_dict = {tmp: vocab})

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, None, None], name="stories")
        self._s_length = tf.placeholder(tf.int32, [None, None, None], name="q_length")
        self._sibs = tf.placeholder(tf.int32, [None, None, None], name = "siblings")
        self._queries = tf.placeholder(tf.int32, [None, None], name="queries")
        self._q_length = tf.placeholder(tf.int32, [None, None], name="q_length")
        self._answers = tf.placeholder(tf.float32, [None, None], name="answers")
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")
        self._dropout = tf.placeholder(tf.float32, [], name="dropout")

    def _build_vars(self, vocab):
        if isinstance(vocab, np.ndarray):
            with tf.variable_scope(self._name):
                tmp = tf.placeholder(tf.float32, vocab.shape)
                self.A_1 = tf.Variable(tmp, name = "dic_A")
        else:
	        with tf.variable_scope(self._name):
	            nil_word_slot = tf.zeros([1, self._embedding_size])
	            A = tf.concat(axis=0, values=[nil_word_slot, self._init(\
	            	[self._vocab_size-1, self._embedding_size]) ])

	            self.A_1 = tf.Variable(A, name="dic_A")

        self._nil_vars = set([self.A_1.name])
        return tmp


    def _inference_rel_attention(self, stories, siblings, queries):
        bs = tf.shape(stories)[0]
        sn = tf.shape(stories)[1]
        sl = tf.shape(self._s_length)[2]
        sbn = tf.shape(siblings)[2]
        #sbl = tf.shape(siblings)[3]
        ql = tf.shape(queries)[1]

        with tf.variable_scope('question_candidates_trans'):
            q_emb = tf.nn.embedding_lookup(self.A_1, queries)
            q_emb = tf.contrib.layers.fully_connected(q_emb, self._embedding_size,
                        None, scope='glove_trans')

            m_emb_story = tf.nn.embedding_lookup(self.A_1, stories)
            m_emb_story = tf.contrib.layers.fully_connected(m_emb_story, self._embedding_size,
                            None, reuse=True, scope='glove_trans')#*remove_story
            #m_emb_story = tf.reshape(m_emb_story, [bs*sn, -1, self._embedding_size])
            # ###
            m_emb_sib = tf.nn.embedding_lookup(self.A_1, siblings)
            m_emb_sib = tf.contrib.layers.fully_connected(m_emb_sib, self._embedding_size,
                            None, reuse=True, scope='glove_trans')


        with tf.variable_scope('question_relation_matching'):
            q_emb = tf.reduce_sum(q_emb, 1)
            m_emb_story = tf.reduce_sum(m_emb_story, 2)

            q_emb = tf.reshape(q_emb, [bs, 1, self._embedding_size])
            m_emb_story_weight = tf.matmul(q_emb, m_emb_story, transpose_b= True)

            probs = tf.nn.softmax(tf.reduce_sum(m_emb_story_weight, 1), -1) #, -1)
            output = tf.stack([probs, probs], 2)

            return probs, probs, output

    def batch_rel_fit(self, stories, sibs, queries, answers, dropout, learning_rate):
        s_length = 1*(stories > 0)
        q_length = 1*(queries > 0)
        feed_dict = {self._stories: stories, self._sibs: sibs, self._queries: queries, \
            self._answers: answers, self._q_length: q_length, self._s_length: s_length, \
            self._dropout: dropout, self._lr: learning_rate}
        loss, pred, show, show1, _ = self._sess.run([self.loss_op, self.predict_op, self.show, \
            self.show1, self.train_op], feed_dict=feed_dict)
        return loss, pred, show, show1

    def predict_rel(self, stories, sibs, queries, answers):
        s_length = 1*(stories > 0)
        q_length = 1*(queries > 0)
        feed_dict = {self._stories: stories, self._sibs: sibs, self._queries: queries, \
            self._answers: answers, self._q_length: q_length, self._s_length: s_length, \
            self._dropout: 1.}
        return self._sess.run([self.predict_op, self.show, self.show1], feed_dict=feed_dict)

    def save_model(self, saver, outfile, tid = 7):
        embedding = self._sess.run(self.A_1, feed_dict={})
        np.save('%s/weights%s' %(outfile, tid), embedding)
        saver.save(self._sess, '%s/model%s.ckpt' %(outfile, tid))

    def obtain_embedding(self):
        return self._sess.run(self.A_1, feed_dict={})

    def obtain_var(self):
        v = [var for var in tf.global_variables() if var.name == \
                'question_candidates_trans/glove_trans/weights:0']
        return self._sess.run(v[0][0, :10], feed_dict={})

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def zero_nil_slot_norm(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot_norm") as name:
        t = tf.norm(t, axis = 1)
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.cast(tf.convert_to_tensor(t, name="t"), tf.float32)
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)
