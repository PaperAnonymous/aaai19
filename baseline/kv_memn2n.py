from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def encoding(sentence_size, embedding_dim):
	encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
	ls = sentence_size + 1
	le = embedding_dim + 1
	for i in range(1, le):
		for j in range(1, ls):
			encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * (j - (sentence_size+1)/2)
	encoding = 1 + 4 * encoding / embedding_dim / sentence_size
	encoding[:, -1] = 1.0
	return np.transpose(encoding)

def zero_nil_slot(t, name=None):
	with tf.op_scope([t], name, "zero_nil_slot") as name:
		t = tf.convert_to_tensor(t, name="t")
		s = tf.shape(t)[1]
		z = tf.zeros(tf.stack([1, s]))
		return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
	with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
		t = tf.convert_to_tensor(t, name="t")
		gn = tf.random_normal(tf.shape(t), stddev=stddev)
		return tf.add(t, gn, name=name)

class KVMemN2N(object):
	def __init__(self, general_config, vocab):

		self._vocab_size = general_config.train_config['voc_sz']
		self._embedding_size = vocab.shape[1]
		self._max_grad_norm = general_config.train_config['max_grad_norm']
		self._hops = general_config.train_config['max_hop']
		self._sentence_size = general_config.train_config['max_words']
		self._config = general_config.train_config
		self._init = tf.random_normal_initializer(stddev=0.1)

		self._build_inputs()
		self._build_vars(vocab)

		self._opt = tf.train.AdagradOptimizer(learning_rate=self._lr)

		# cross entropy
		logits = self._inference(self._stories, self._queries, self._sibs, self._sib_length)
		#cross_entropy_sum = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
		#	labels=tf.cast(self._answers, tf.float32), name='cross_entropy')
		# cross_entropy_sum = kl_divergence(self._answers, logits)
		cross_entropy_sum = tf.reduce_sum((self._answers - logits)**2)

		# loss op
		loss_op = cross_entropy_sum

		# gradient pipeline
		grads_and_vars = self._opt.compute_gradients(loss_op)
		grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
		grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
		nil_grads_and_vars = []
		for g, v in grads_and_vars:
			if v.name in self._nil_vars:
				nil_grads_and_vars.append((zero_nil_slot(g), v))
			else:
				nil_grads_and_vars.append((g, v))
		train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

		# predict ops
		predict_op = logits
		predict_proba_op = logits
		predict_log_proba_op = logits

		# assign ops
		self.loss_op = loss_op
		self.predict_op = predict_op
		self.predict_proba_op = predict_proba_op
		self.predict_log_proba_op = predict_log_proba_op
		self.train_op = train_op

		init_op = tf.global_variables_initializer()
		self._sess = tf.Session()
		self._sess.run(init_op)

	def _build_inputs(self):
		self._stories = tf.placeholder(tf.int32, [None, None, None], name="stories")
		self._queries = tf.placeholder(tf.int32, [None, None], name="queries")
		self._sibs = tf.placeholder(tf.int32, [None, None, None], name="sibs")
		self._answers = tf.placeholder(tf.float32, [None, None], name="answers")
		self._lr = tf.placeholder(tf.float32, [], name="learning_rate")
		self._sib_length = tf.placeholder(tf.float32, [None, None], name = "sibs_length")

	def _build_vars(self, vocab):
		with tf.variable_scope('memn2n'):
			nil_word_slot = tf.zeros([1, self._embedding_size])
			A = tf.concat(axis=0, values=[ nil_word_slot, 
				self._init([self._vocab_size-1, self._embedding_size]) ])
			B = tf.constant(vocab.astype(np.float32))

			self.A_1 = tf.get_variable("A", initializer = vocab)

			self.R = []

			for hopn in range(self._hops):
				with tf.variable_scope('hop_{}'.format(hopn)):
					self.R.append(tf.get_variable("R", 
						shape=[self._embedding_size, self._embedding_size]))

		self._nil_vars = set([self.A_1.name] + [x.name for x in self.R])

	def _inference(self, stories, queries, sib, sib_length):
		sib_length = tf.expand_dims(sib_length, -1)
		with tf.variable_scope('memn2n'):
			# Use A_1 for thee question embedding as per Adjacent Weight Sharing
			q_r = tf.reduce_sum(tf.nn.embedding_lookup(self.A_1, queries), 1)
			doc_r = tf.reduce_sum(tf.nn.embedding_lookup(self.A_1, stories), 2)
			value_r = tf.reduce_sum(tf.nn.embedding_lookup(self.A_1, sib), 2)*sib_length

			u = [q_r]
			memory_probs = []

			for hopn in range(self._hops):
				R = self.R[hopn]

				u_expanded = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                # hack to get around no reduce_dot
				dotted = tf.reduce_sum(doc_r * u_expanded, 2)

				# Calculate probabilities
				probs = tf.nn.softmax(dotted)
				memory_probs.append(probs)

				probs_expanded = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])

				v_temp = tf.transpose(value_r, [0, 2, 1])

				o_k = tf.reduce_sum(v_temp * probs_expanded, 2)

				u_k = tf.matmul(R, u[-1] + o_k, transpose_b = True)

				u.append(tf.transpose(u_k))

            # Use last C for output (transposed)
			o = u[-1]
			value_r = tf.nn.embedding_lookup(self.A_1, sib)
			value_r = tf.reduce_sum(value_r, 2)*sib_length
			logits = tf.matmul(tf.expand_dims(o, 1), value_r, transpose_b = True)
			logits = tf.nn.softmax(tf.squeeze(logits, 1), 1)
			return logits

	def batch_rel_fit(self, stories, sib, queries, answers, dropout, lrate):
		sib_length = 1./np.maximum(1, np.sum(1.*(sib > 0), 2))
		#print(sib_length)
		feed_dict = {self._stories: stories, self._queries: queries, self._sibs: sib, 
			self._answers: answers, self._lr: lrate, self._sib_length: sib_length}
		loss, prob, _ = self._sess.run([self.loss_op, self.predict_proba_op, 
			self.train_op], feed_dict=feed_dict)

		return loss, prob, prob, prob

	def predict_rel(self, stories, sib, queries, answers):
		sib_length = 1./np.maximum(1, np.sum(1.*(sib > 0), 2))
		feed_dict = {self._stories: stories, self._queries: queries, self._sibs: sib, 
			self._sib_length: sib_length}
		prob = self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

		return prob, prob, prob

	def obtain_var(self):
		v = [var for var in tf.global_variables() if var.name == 'memn2n/A:0']
		return self._sess.run(v[0][0, :10], feed_dict={})

	def obtain_embedding(self):
		return self._sess.run(self.A_1, feed_dict={})

	def save_model(self, saver, outfile, tid = 7):
		embedding = self._sess.run(self.A_1, feed_dict={})
		np.save('%s/weights%s' %(outfile, tid), embedding)
		saver.save(self._sess, '%s/model%s.ckpt' %(outfile, tid))