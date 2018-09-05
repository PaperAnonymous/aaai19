import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import time

torch.manual_seed(123)

def BiRNN(sequence, n_h, sequence_w_len = None, reuse = None, keep_prob = 0.8,
		scope = None):
    cell_fw = DropoutWrapper(tf.contrib.rnn.LSTMCell(n_h, reuse=reuse),
                             output_keep_prob=keep_prob,
                             dtype=tf.float32)
    cell_bw = DropoutWrapper(tf.contrib.rnn.LSTMCell(n_h, reuse=reuse),
                             output_keep_prob=keep_prob,
                             dtype=tf.float32)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                            sequence, sequence_length=sequence_w_len,
                                            dtype=tf.float32,
                                            scope = scope)
    return tf.concat(outputs, 2)

def foward_ReLU(sequence, n_h, reuse = None, keep_prob = 1., scope = None):
    sequence = tf.nn.dropout(sequence, keep_prob)
    output = tf.contrib.layers.fully_connected(sequence, n_h, \
                                               tf.nn.relu, reuse = reuse,
                                               scope = scope)
    return output

def RNN(sequence, n_h, sequence_w_len = None, reuse = None, keep_prob = 0.8,
		scope = None):
    cell = DropoutWrapper(tf.contrib.rnn.LSTMCell(n_h, reuse=reuse),
                          output_keep_prob=keep_prob,
                          dtype=tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, sequence,
                                    sequence_length=sequence_w_len,
                                    dtype=tf.float32,
                                    scope = scope)
    return outputs

def kl_divergence(p, q):
    return p * tf.log(tf.maximum(1.e-10, p/q))

class Z_Layer(nn.Module):
	def __init__(self, n_in, n_hidden = 30, dropout = 0.8):
		super(Z_Layer, self).__init__()
		self._n_hidden = n_hidden
		self._n_in = n_in

		# self.w1 = Variable(torch.randn(2*n_in, 1).type(torch.FloatTensor)
		# 					, requires_grad=True)
		# self.w2 = Variable(torch.randn(n_hidden, 1).type(torch.FloatTensor)
		# 					, requires_grad=True)
		# self.bias = Variable(torch.randn(1).type(torch.FloatTensor)
		# 					, requires_grad=True)
		self.w21 = Variable(torch.randn(n_in, 1).type(torch.FloatTensor)
							, requires_grad=True)
		self.w22 = Variable(torch.randn(n_hidden, 1).type(torch.FloatTensor)
							, requires_grad=True)
		self.bias2 = Variable(torch.randn(1).type(torch.FloatTensor)
							, requires_grad=True)
		# self.w31 = Variable(torch.randn(n_in, 1).type(torch.FloatTensor)
		# 					, requires_grad=True)
		# self.w32 = Variable(torch.randn(n_hidden, 1).type(torch.FloatTensor)
		# 					, requires_grad=True)
		# self.bias3 = Variable(torch.randn(1).type(torch.FloatTensor)
		# 					, requires_grad=True)
		#self.rlayer1 = nn.LSTMCell(2*n_in+1, n_hidden)
		self.rlayer2 = nn.LSTMCell(n_in+1, n_hidden)
		#self.rlayer3 = nn.LSTMCell(n_in+1, n_hidden)

	def sample_unit(self, x, prev):
		z_tm1, h_tm1, c_tm1 = prev
		w1, w2, bias = self.w21, self.w22, self.bias2

		B, n_in = x.size()
		pz_t = torch.sigmoid(torch.matmul(x, w1) + torch.matmul(h_tm1, w2) + bias)
		#print(torch.matmul(h_tm1, w2).size())

		hz_t = torch.cat((x, pz_t), 1)

		h_t, c_t = self.rlayer2(hz_t, (h_tm1, c_tm1))
		return [pz_t, h_t, c_t]

	def sample_all(self, h_concat, dep_hc = None):
		B, QL, n_in = h_concat.size()
		n_hidden = self._n_hidden
		h = Variable(torch.zeros(B, n_hidden))
		c = Variable(torch.zeros(B, n_hidden))
		z = Variable(torch.zeros(B, ))
		# _h = Variable(torch.zeros(B, n_hidden))
		# _c = Variable(torch.zeros(B, n_hidden))
		# _z = Variable(torch.zeros(B, ))
		Z = Variable(torch.zeros(B, QL))
		# _Z = Variable(torch.zeros(B, QL))

		for t in range(h_concat.size(1)):
			x = h_concat[:, t, :]
			_x = h_concat[:, h_concat.size(1)-t-1, :]
			z, h, c = self.sample_unit(x, (z, h, c))
			#_z, _h, _c = self._sample_unit(_x, (_z, _h, _c))
			# if dep_hc is None:
			# 	z = 1.
			Z[:, t] = torch.squeeze(z)
			#_Z[:, h_concat.size(1)-t-1] = _z
			#Z = (Z + _Z)/2.
		return Z, True

	def sample_unit2(self, i, x, i_prev, x_prev):
		i_z_tm1, i_h_tm1, i_c_tm1 = i_prev
		x_z_tm1, x_h_tm1, x_c_tm1 = x_prev
		i_w1, i_w2, i_bias = self.w1, self.w2, self.bias
		x_w1, x_w2, x_bias = self.w21, self.w22, self.bias2

		B, n_in = i.size()
		i_pz_t = torch.sigmoid(torch.matmul(i, i_w1) + torch.matmul(i_h_tm1, i_w2) + i_bias)
		x_pz_t = torch.sigmoid(torch.matmul(x, x_w1) + torch.matmul(x_h_tm1, x_w2) + x_bias)

		i_hz_t = torch.cat((i, i_pz_t), 1)
		x_hz_t = torch.cat((x, x_pz_t), 1)

		i_h_t, i_c_t = self.rlayer1(i_hz_t, (i_h_tm1, i_c_tm1))
		x_h_t, x_c_t = self.rlayer2(x_hz_t, (x_h_tm1, x_c_tm1))
		return (i_pz_t, i_h_t, i_c_t), (x_pz_t, x_h_t, x_c_t)

	def sample_all2(self, dep, diff, dep_hc = None):
		B, QL, n_in = diff.size()
		n_hidden = self._n_hidden
		if dep_hc is None:
			dep_h = Variable(torch.zeros(B, n_hidden))
			dep_c = Variable(torch.zeros(B, n_hidden))
		else:
			dep_h, dep_c = dep_hc
		diff_h = Variable(torch.zeros(B, n_hidden))
		diff_c = Variable(torch.zeros(B, n_hidden))
		dep_z = Variable(torch.zeros(B, ))
		diff_z = Variable(torch.zeros(B, ))
		Z = Variable(torch.zeros(B, QL))

		for t in range(diff.size(1)):
			x = diff[:, t, :]
			i = dep[:, t, :]
			(dep_z, dep_h, dep_c), (diff_z, diff_h, diff_c) = \
				self.sample_unit2(i, x, (dep_z, dep_h, dep_c), (diff_z, diff_h, diff_c))
			if dep_hc is None:
				diff_z = 1.
			Z[:, t] = (dep_z + diff_z)/2.
		return Z, (dep_h, dep_c)

torch.manual_seed(123)

class Ranker(nn.Module):
	def __init__(self, args, emb, lr = 0.1, dropout = 0.8):
		super(Ranker, self).__init__()
		self.args = args
		self.emb = emb
		self.vocab_size, self.emb_dim = emb.shape
		n_d = args.hidden_dimension

		self.ACTIVATION_DICT = {'tanh':nn.Tanh,
                                 'sigmoid':nn.Sigmoid}

		self.emb_init = nn.Embedding(self.emb.shape[0], self.emb.shape[1])
		self.emb_init.weight.data.copy_(torch.from_numpy(self.emb))
		self.encode_rnn = nn.LSTM(self.emb_dim, n_d/2, dropout = dropout,
								batch_first=True, bidirectional = True)
		self.Z_layer = Z_Layer(n_d, n_hidden = n_d, dropout = dropout)
		self.compare_rnn = nn.LSTM(2*n_d, n_d, dropout = dropout, batch_first=True)
		self.end_rnn = nn.LSTM(2*n_d, n_d, dropout = dropout, batch_first= True)
		self.Linear_layers = nn.Linear(n_d, 1)
		self.end_layers = nn.LSTM(n_d, 1)

		self.lr = lr
		self.dropout = dropout

	def forward(self, x, s):
		padding_id = self.padding_id = 0
		n_d = self.args.hidden_dimension
		n_e = self.emb_dim
		B, SN, SL = s.size()
		B, QL = x.size()

		self.emb_init.weight.data = F.normalize(self.emb_init.weight.data, p=2, dim=1)
		self.emb_init.weight.data[0, :].fill_(0)

		inputs = self.inputs = self.emb_init(x)
		story = self.emb_init(s.contiguous().view(-1, SL)).view(B, SN, SL, -1)

		activation = self.ACTIVATION_DICT[self.args.activation]

		h0 = Variable(torch.zeros(2, B, n_d/2))
		c0 = Variable(torch.zeros(2, B, n_d/2))
		self.inputs = inputs = self.encode_rnn(inputs, (h0, c0))[0]
		story = story.view(B*SN, SL, n_e)
		h0 = h0.repeat(1, SN, 1)
		c0 = c0.repeat(1, SN, 1)
		story = self.encode_rnn(story, (h0, c0))[0]
		self.story = story.contiguous().view(B, SN, SL, n_d)

		trans_x = torch.transpose(inputs, 2, 1)
		trans_story = story.view(B, SN*SL, n_d)
		self.alig = alig = torch.matmul(trans_story, trans_x)

		self.x_masks = x_masks = 1 - torch.eq(x, padding_id).type(torch.FloatTensor)
		self.story_masks = story_masks = 1 - torch.eq(s, padding_id).type(torch.FloatTensor)

		# s_masks = self.story_masks.view(B, -1).unsqueeze(2)
		# x_masks = self.masks.unsqueeze(1)
		# mask_alig = torch.matmul(s_masks, x_masks)

		# mask_alig_values = -1e10*torch.ones_like(alig)
		# alig = mask_alig*alig + (1-mask_alig)*mask_alig_values

		# alig1 = alig.view(B*SN, SL, QL)
		# alig1 = F.softmax(alig1, 1)
		# self.mask_alig = alig1.view(B, SN, SL, QL)
		# mask_alig_values = 0.*torch.ones_like(alig1)
		# mask_alig = mask_alig.view(B*SN, SL, QL)
		# self.alig1 = alig1 = mask_alig*alig1 + (1-mask_alig)*mask_alig_values

		# soft_input = torch.matmul(torch.transpose(alig1, 2, 1), story)
		# soft_input = soft_input.view(B, SN, QL, n_d)
		# self.compare = compare = torch.mean(soft_input, 1)

		# self.output_layer = output_layer = self.Z_layer
		# probs = output_layer.sample_all(compare)
		# self.probs = probs

		# self.zsum = torch.sum(probs, 1)
		# self.zdiff = torch.sum(torch.abs(probs[:, 1:] - probs[:, :-1]), 1)
		# self.zin = torch.sum((1 - probs)*e, 1)

		story_masks1 = story_masks.view(B, -1)
		s_masks = torch.unsqueeze(story_masks1, 2)
		z_masks = torch.unsqueeze(x_masks, 1)
		mask_alig = torch.matmul(s_masks, z_masks)

		mask_alig_values = -1e10*torch.ones_like(alig)
		alig = mask_alig*alig + (1-mask_alig)*mask_alig_values
		alig = F.softmax(alig, 2)

		mask_alig_values = 0.*torch.ones_like(alig)
		self.alig = alig = mask_alig*alig + (1-mask_alig)*mask_alig_values

		self.soft_story = soft_story = torch.matmul(alig, inputs)

		s_story = story.view(B, -1, n_d)
		self.s_story = s_story
		compare = torch.cat((s_story * soft_story,
							(s_story - soft_story)**2), 2)
		compare = compare.view(B*SN, SL, 2*n_d)
		h0 = Variable(torch.zeros(1, B*SN, n_d))
		c0 = Variable(torch.zeros(1, B*SN, n_d))
		compare_weight, _ = self.compare_rnn(compare, (h0, c0))
		self.compare_weight = compare_weight = torch.max(compare_weight, 1)[0]
		logits = self.Linear_layers(compare_weight)
		self.logits = logits = logits.view(B, SN)
		self.preds = preds = F.softmax(logits, -1)

		end_weight, _ = self.end_rnn(compare, (h0, c0))
		self.end_weight = end_weight = torch.max(end_weight, 1)[0]
		self.end_signal = end_signal = self.end_layers(end_weight)

		return preds, end_signal

	def obtain_reward(self, y, preds):
		loss_mat = self.loss_mat = F.kl_div(preds, y, size_average=True, reduce=False)
		#loss_mat = self.loss_mat = (y - preds)**2

		# self.zsum = zsum = gen.zsum
		# self.zdiff = zdiff = gen.zdiff
		# self.zin = zin = gen.zin
		#print(logpz)
		self.loss_vec = loss_vec = torch.sum(loss_mat, 1) #+ zsum*args.sparsity + \
								#zdiff*args.coherent + zin*args.zin
		self.loss = loss = torch.sum(loss_vec)

		self.cost_g = loss
		return loss

class Ranker2(nn.Module):
	def __init__(self, args, emb, lr = 0.1, dropout = 0.2):
		super(Ranker2, self).__init__()
		self.args = args
		self.emb = emb
		self.vocab_size, self.emb_dim = emb.shape
		n_d = args.hidden_dimension

		self.ACTIVATION_DICT = {'tanh':nn.Tanh,
                                 'sigmoid':nn.Sigmoid}

		self.emb_init = nn.Embedding(self.emb.shape[0], self.emb.shape[1])
		self.emb_init.weight.data.copy_(torch.from_numpy(self.emb))
		self.encode_rnn = nn.LSTM(n_d, int(n_d/2), dropout = 0.,
								batch_first=True, bidirectional = True)
		self.encode_linear= nn.Sequential(nn.Linear(self.emb_dim, n_d), nn.ReLU())
		#self.s_encode_linear= nn.Linear(self.emb_dim, n_d)
		#self.match_layers = nn.Linear(n_d, 1)
		in_dim = (n_d + n_d) if args.with_dep else n_d
		self.Z_layer = Z_Layer(n_d, n_hidden = n_d, dropout = dropout)
		#self.Z_layer2 = Z_Layer(n_d, n_hidden = n_d, dropout = dropout)
		self.compare_rnn1 = nn.LSTM(n_d, n_d, dropout = dropout, batch_first=True)
		#self.compare_rnn2 = nn.LSTM(n_d, n_d, dropout = dropout, batch_first=True)
		#self.compare_cnn = nn.Conv1d(2*n_d, n_d, 3, stride=1, padding = 1)
		#self.end_rnn = nn.LSTM(2*n_d, n_d, dropout = dropout, batch_first= True)
		self.Linear_layers = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(n_d, 1))
		#self.trans_layers = nn.Sequential(nn.Linear(n_d, n_d), nn.Tanh())
		self.compare_layers = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(2*n_d+1, n_d))
		#self.stop_layers = nn.Sequential(nn.Linear(n_d, 1), nn.Sigmoid())
		self.Z_linear = nn.Sequential(nn.Linear(n_d, 1), nn.Sigmoid())

		self.lr = lr
		self.dropout = dropout

	def encoder(self, x):
		padding_id = self.padding_id = 0
		B, QL = x.size()
		n_d = self.args.hidden_dimension
		#cells = Variable(torch.zeros(B, QL, n_d), requires_grad=False)

		time1 = time.time()
		#print('normalise time %s' %(time.time() - time1))
		#print(torch.sum(self.emb_init.weight.data[10, :]**2))
		self.emb_init.weight.data[0, :].fill_(0)
		#print('fill 0 %s' %(time.time() - time1))
		time1 = time.time()

		inputs = self.inputs = self.emb_init(x)
		self.x_masks = x_masks = 1 - torch.eq(x, padding_id).type(torch.FloatTensor)

		#h0 = Variable(torch.zeros(1, B, n_d))
		#c0 = Variable(torch.zeros(1, B, n_d))
		self.inputs = inputs = self.encode_linear(inputs)
		#self.inputs = inputs = self.encode_rnn(inputs)[0].contiguous()
		inputs = inputs.view(B, -1, QL, n_d)
		#print('input view %s' %(time.time() - time1))
		return inputs

	def ranker(self, inputs, s, t, dep_hc, dep = None):
		# print('s %s' %str(s.size()))
		# print('defore input %s' %str(inputs.size()))
		padding_id = self.padding_id = 0
		n_d = self.args.hidden_dimension
		n_e = self.emb_dim
		B, SN, SL = s.size()
		_, top, QL, _ = inputs.size()

		story = self.emb_init(s.contiguous().view(-1, SL)).view(-1, SN, SL, n_e)
		story = story.view(-1, SL, n_e)
		story = self.encode_linear(story)
		#story = self.encode_rnn(story)[0].contiguous()
		story = story.view(B,1,SN,SL,n_d).repeat(1,top,1,1,1).view(-1,SL,n_d)
		inputs = inputs.view(-1, QL, n_d)

		trans_x = torch.transpose(inputs, 1, 2)
		trans_story = story.view(-1, SN*SL, n_d)
		alig = torch.matmul(trans_story, trans_x).view(-1, SL, QL)
		#print(alig[0, :, :])

		story_masks = 1 - torch.eq(s, padding_id).type(torch.FloatTensor)

		story_masks1 = story_masks.view(-1, SN*SL)
		s_masks = torch.unsqueeze(story_masks1, 2)
		z_masks = torch.unsqueeze(self.x_masks, 1)
		mask_alig = torch.matmul(s_masks, z_masks).view(B, SN, SL, QL)
		mask_alig = torch.unsqueeze(mask_alig,1).repeat(1,top,1,1,1).view(-1,SL,QL)

		mask_alig_values = -1e10*torch.ones_like(alig)
		alig = mask_alig*alig + (1-mask_alig)*mask_alig_values
		alig1 = F.softmax(alig, 1)
		#print(alig1)
		alig2 = F.softmax(alig, 2) #coverage !!!

		mask_alig_values = 0.*torch.ones_like(alig)
		alig1 = mask_alig*alig1 + (1-mask_alig)*mask_alig_values
		alig1 = torch.transpose(alig1, 1, 2)
		alig2 = mask_alig*alig2 + (1-mask_alig)*mask_alig_values #coverage !!!

		soft_inputs = torch.matmul(alig1, story).view(-1, SN, QL, n_d)
		#inputs_tmp = inputs.view(-1,1,QL,n_d).repeat(1,SN,1,1).view(-1,QL,n_d)
		#soft_story = torch.matmul(alig2, inputs_tmp).view(-1, SN, SL, n_d)

		s_inputs = inputs.view(-1, 1, QL, n_d).repeat(1, SN, 1, 1)
		# i_story = story.view(-1, SN, SL, n_d)

		alig2 = torch.sum(alig2, 1).view(-1, SN, QL, 1) #coverage!!!
		dep_hc = dep_hc.view(-1, 1, QL, 1).repeat(1, SN, 1, 1).view(-1, SN, QL, 1) # coverage !!!
		dep_hc = dep_hc + alig2 # converage
		compare1 = torch.cat((s_inputs * soft_inputs,
							(s_inputs - soft_inputs)**2,
							 dep_hc), 3)
		compare1 = compare1.view(-1, QL, 2*n_d+1)
		#compare1 = torch.transpose(compare1, 2, 1)
		compare1 = self.compare_layers(compare1)
		compare1, _ = self.compare_rnn1(compare1)
		#compare1 = self.compare_cnn(compare1)
		#compare1 = torch.transpose(compare1, 2, 1)
		#inputs = compare1.contiguous().view(B, -1, QL, n_d)
		#inputs = inputs.view(-1, 1, QL, n_d).repeat(1, SN, 1, 1).view(B, -1, QL, n_d) #coverage!!!
		dep_hc = dep_hc.view(B, -1, QL, 1)
		compare_weight1 = torch.max(compare1, 1)[0]

		#compare2 = torch.cat((i_story * soft_story,
		#					(i_story - soft_story)**2), 3)
		#compare2 = compare2.view(-1, SL, 2*n_d)
		#compare2 = self.compare_layers(compare2)
		#compare2, _ = self.compare_rnn2(compare2)
		#compare_weight2 = torch.max(compare2, 1)[0]

		#compare_weight1 = torch.cat((compare_weight1, compare_weight2), 1)

		logits = self.Linear_layers(compare_weight1)
		logits = logits.view(B, -1, SN)
		preds = F.softmax(logits, -1)
		probs = torch.squeeze(self.Z_linear(compare_weight1), -1).view(-1, SN)
		inputs = inputs.view(-1, 1, QL, n_d).repeat(1, SN, 1, 1)
		#inputs = inputs.view(-1, QL, n_d)*torch.unsqueeze(probs, -1)
		inputs = inputs.view(B, -1, QL, n_d)
		#print(probs)
		stop = 1. - torch.min(probs)
		'''
		#remark
		compare1 = torch.cat((s_inputs * soft_inputs,
							(s_inputs - soft_inputs)**2), 3)
		compare1 = F.relu(self.compare_layers(compare1))
		compare1 = compare1.view(-1, QL, n_d)
		compare1, _ = self.compare_rnn1(compare1)
		self.compare_weight1 = compare_weight1 = torch.max(compare1, 1)[0]

		logits = self.Linear_layers(compare_weight1)
		logits = logits.view(B, -1, SN)
		self.preds = preds = F.softmax(logits, -1)
		#print(preds)

		match_inputs = soft_inputs
		inputs = inputs.view(-1, 1, QL, n_d).repeat(1, SN, 1, 1)
		diff = (inputs - match_inputs)**2
		diff = diff.view(-1, QL, n_d)

		if dep is not None:
			#print(dep[0, :, :])
			if isinstance(dep, tuple):
				dep1, dep2 = dep
				dep1 = self.emb_init(dep1.contiguous().view(B*QL, -1)).view(B, QL, -1, n_e)
				dep1 = self.encode_linear(dep1)
				dep1 = torch.sum(dep1, 2)
				dep2 = self.emb_init(dep2.contiguous().view(B, QL)).view(B, QL, n_e)
				dep2 = self.encode_linear(dep2)
				dep = torch.cat((dep1, dep2), 2)
			elif dep.dim == 3:
				dep = self.emb_init(dep.contiguous().view(B*QL, -1)).view(B, QL, -1, n_e)
				dep = self.encode_linear(dep)
				dep = torch.sum(dep, 2)
			elif dep.dim == 2:
				dep = self.emb_init(dep.contiguous().view(B, QL)).view(B, QL, n_e)
				dep = self.encode_linear(dep)
		#	diff = torch.cat((diff, dep), 2)
			probs, dep_hc = self.Z_layer.sample_all2(dep, diff, dep_hc = dep_hc)
		else:
			probs, dep_hc = self.Z_layer.sample_all(diff, dep_hc = dep_hc)
			#probs = torch.squeeze(self.Z_linear(diff), -1)
		inputs = inputs.view(-1, QL, n_d)*torch.unsqueeze(probs, -1)
		# print(probs)
		stop = 1. - torch.min(torch.mean(probs, -1))
		inputs = inputs.view(B, -1, QL, n_d)
		'''
		return preds, inputs, stop, dep_hc

	def final_probs(self, preds, prev_preds, s):
		B = preds.size(0)

		s = torch.sum(s, 2)
		mask_alig = 1 - torch.eq(s, self.padding_id).type(torch.FloatTensor)
		mask_alig_values = 0*torch.ones_like(mask_alig)

		#print(prev_preds)
		prev_preds = torch.unsqueeze(prev_preds, 2)
		#print(preds)
		#preds = torch.unsqueeze(preds, 1)
		preds = (prev_preds*preds).view(B, -1)

		preds = mask_alig*preds + (1-mask_alig)*mask_alig_values

		return preds

	def obtain_reward(self, y, preds, step_loss):

		#print(preds.size())
		#loss_mat = self.loss_mat = F.kl_div(preds, y, size_average=False, reduce=True)
		#print(preds)
		loss_mat = self.loss_mat = torch.sum((y - preds)**2)
		print('step loss %s\t loss mat %s' %(step_loss.data.numpy(), loss_mat.data.numpy()))
		self.loss = loss = loss_mat + step_loss
		return loss
