import numpy as np
import sys
import gzip
import random
import json
import re
from collections import defaultdict
import zipfile
import time
import os
#import editdistance

np.random.seed(2345)

global qtype
qtype = set(['starred actors', 'directed by', 'has genre', 'in language', 'has tags',
	'written by', 'release year', 'has imdb rating', 'has imdb votes'])
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
	'yers','wrider','be','riter','whos','wended','this','you','right','state'])
global split2hop
split2hop = {2: 1, 3: 1, 5: 2, 7:3}

def create_embedding_layer(path):
    embedding_layer = EmbeddingLayerTf(
            n_d = 200,
            vocab = [ "<unk>", "<padding>" ],
            embs = load_embedding_iterator(path),
            oov = "<unk>",
            #fix_init_embs = True
            fix_init_embs = False
        )
    return embedding_layer

def read_annotations(path, dic, is_WEBQA = False):
	question, answer, entity_pos = [], [], []
	count = defaultdict(int)
	q_len = 0
	if is_WEBQA:
		with open(path) as f:
			for line_idx, line in enumerate(f):
				q2idx, a2idx = (), ()
				line = line.replace('\n', '').lower()
				line = re.sub("'s", " 's", line)
				line = re.sub("\?$", '', line)
				line = re.sub("[ ]+", " ", line)
				q_txt = line.split('\t')[-1]
				a_txt = line.split('\t')[:-1]

				q_txt = re.split('\_| ', q_txt)
				for i, w in enumerate(q_txt):
					if w not in dic:
						dic[w] = len(dic)
					q2idx += (dic[w], )

				for i, w in enumerate(a_txt):
					if re.search('\d+/\d+/\d+', w) and not re.search('[a-z]+', w):
						d, m, y = w.split('/')
						w = '-'.join([y, m.zfill(2), d.zfill(2)])
					a2idx += (w, )

				question += [q2idx]
				answer += [a2idx]

				if q_len < len(q2idx):
					q_len = len(q2idx)
		with open(re.sub('_', '_mid_', path), 'r') as f:
			for line_idx, line in enumerate(f):
				line = line.replace('\n', '').lower()
				line = re.sub("'s", " 's", line)
				entities = line.split('\t')
				entity_pos += [entities]
	else:
		with open(path) as f:
			for line_idx, line in enumerate(f):
				q2idx, a2idx = (), ()
				line = line.replace('\n', '').lower()
				line = re.sub("'s", " 's", line)
				line = re.sub("[ ]+", " ", line)
				line = re.sub("\_", " ", line)
				q_txt, a_txt = line.split('\t')

				q_txt = re.sub('(?<=\]),', '', q_txt)
				q_txt = q_txt.strip()
				topic_entity = [not not re.search('[\[\]]', w) for w in q_txt.split(' ')]
				entity_pos += [tuple([i for i, w in enumerate(topic_entity) if w])]
				q_txt = re.sub('[\[\]]', '', q_txt)
				q_txt = re.split(' ', q_txt)
				for i, w in enumerate(q_txt):
					if w not in dic:
						dic[w] = len(dic)
					q2idx += (dic[w], )
					count[w] += 1

				a_txt = a_txt.split('|')
				for i , w in enumerate(a_txt):
					a2idx += (w, )

				question += [q2idx]
				answer += [a2idx]
	# count = sorted(count.items(), key = lambda (k, v): v, reverse = True)[:50]
	# print(count)
	# print("','".join([w for w, _ in count]))
	# stop
	# if isinstance(list(stop_words)[0], str):
	# 	global stop_words
	# 	stop_words = set([dic[w] for w in stop_words])
	return question, answer, entity_pos

def read_kb(path, dic, is_WEBQA = False):
	sub_idx = defaultdict(list)
	kb = defaultdict(list)
	max_line_idx, max_k = 0, 0
	open_tool = open(path) if not is_WEBQA else gzip.open(path)
	with open_tool as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '').lower()
			words = line.split('\t') if is_WEBQA else line.split('|')
			#print(words)
			for i in [0, 1, 2]:
				sub = ()
				word = words[i]

				if i in [0, 2]:
					word = re.sub("'s", " 's", word)
					if re.search('\d+/\d+/\d+', word) and not re.search('[a-z]+', word):
						d, m, y = word.split('/')
						word = '-'.join([y, m.zfill(2), d.zfill(2)])
					word = word.split(' ')
				elif i == 1:
					word = re.split('\_|[ ]+', word)

				text = ()
				for k, w in enumerate(word):
					if w not in dic:
						dic[w] = len(dic)
					text += (dic[w], )
					sub += (dic[w], )
				kb[line_idx] += [text]

				if is_WEBQA:
					if i == 0:
						sub_idx[sub] += [line_idx]
				else:
					if i == 0:
						sub_idx[sub] += [(1, line_idx)]
					elif i == 2:
						sub_idx[sub] += [(-1, line_idx)]

				if max_k < k + 1:
					max_k = k + 1

			if line_idx % 10000000 == 0:
				print('read kb ... %s' %line_idx)
			# if line_idx == 100000:
			# 	break

		if max_line_idx < line_idx + 1:
			max_line_idx = line_idx + 1
	return kb, sub_idx

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

def read_golden_rel(path, is_tranform = False):
	goldens = []
	goldennums = []
	qtype2rel = {'actor_to_movie': 'starred actors -1',
	'director_to_movie': 'directed by -1', 'movie_to_actor': 'starred actors',
	'movie_to_director': 'directed by', 'movie_to_genre': 'has genre',
	'movie_to_language': 'in language', 'movie_to_tags': 'has tags',
	'movie_to_writer': 'written by', 'movie_to_year': 'release year',
	'tag_to_movie': 'has tags -1', 'writer_to_movie': 'written by -1',
	'movie_to_imdbrating': 'has imdb rating', 'movie_to_imdbvotes': 'has imdb votes'}
	if is_tranform:
		category = re.findall('(?<=qa_)[^\_]+', os.path.basename(path))[0]
		topics = []
		with open('./data/mix-hop/vanilla/qa_%s.txt' %category) as f:
			for line_idx, line in enumerate(f):
				line = line.strip().lower()
				topic = re.findall('(?<=\[)[^\]]+', line)
				if line_idx == 4263:
					print(line)
				topics += topic
	with open(path) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '').lower()
			golden = []
			if is_tranform:
				line = line.split('_')
				for i in range(int(np.ceil(len(line)/3.))):
					golden += [qtype2rel['_'.join(line[i*2: i*2+3])]]
				golden_num = len(golden)
				golden = topics[line_idx] + ' ' + ' '.join(golden)
			else:
				golden_num = int(len(re.findall('#', line))/2)
				line = '#'.join([w for i, w in enumerate(line.split('#')) if i not in [2, 4, 6]])
				golden = re.sub('[\_|\#]+', ' ', line)
			goldens += [golden]
			goldennums += [golden_num]
	return (goldennums, goldens)

def read_dep(path, dep_type = 'weight', dic = None):
	deps = []
	if dep_type == 'weight':
		with open(path) as f:
			for line in f:
				line = line.replace('\n', '')
				idx, dep_str = line.split('\t')
				dep = ()
				for d in dep_str.split('   '):
					#print(d)
					dep += (int(d), )
				deps += [dep]
	elif dep_type == 'tag':
		with open(path) as f:
			for line in f:
				line = line.replace('\n', '')
				idx, dep_str = line.split('\t')
				dep = ()
				for d in dep_str.split(' '):
					if dic and d not in dic:
						dic[d] = len(dic)
					dep += (dic[d], )
				deps += [dep]
	elif dep_type == 'path':
		with open(path) as f:
			for line in f:
				line = line.replace('\n', '')
				idx, dep_str = line.split('\t')
				dep = ()
				for ds in dep_str.split(' '):
					one_dep = ()
					for d in ds.split('|'):
						if dic and d not in dic:
							dic[d] = len(dic)
						one_dep += (dic[d], )
					dep += (one_dep, )
				deps += [dep]
	return deps

def read_alias(path):
	alias = defaultdict(set)

	with open(path, 'r') as f:
		for line in f:
			line = line.replace('\n', '')
			line = line.split('\t')

			for i in range(len(line)):
				words = line[i]

				if i == 0:
					head = words
				else:
					alia = words
					if head != alia:
						alias[alia].add(head)
	return alias

def save_config(config, path):
	with open(path, 'w') as f:
	    for key, value in config.__dict__.items():
	        f.write(u'{0}: {1}\n'.format(key, value))

def save_kbsubidx(path, kb, sub_idx):
	f = gzip.open(path, 'w')
	for line_idx in kb:
		text = '%s\t%s\t%s\t%s' %(str(line_idx), ' '.join(map(str, kb[line_idx][0])),
			' '.join(map(str, kb[line_idx][1])), ' '.join(map(str, kb[line_idx][2])))
		f.write(text + '\n')
	f.write('***\n')
	for sub in sub_idx:
		text = '%s\t%s' %(' '.join(map(str, sub)), '\t'.join(map(str, sub_idx[sub])))
		f.write(text + '\n')
	f.close()

def load_kbsubidx(path):
	kb = {}
	sub_idx = {}
	is_subidx = False
	with gzip.open(path, 'r') as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')
			if line == '***':
				is_subidx = True
			else:
				if is_subidx:
					sub = tuple(map(int, line.split('\t')[0].split(' ')))
					idx = map(int, line.split('\t')[1:])
					sub_idx[sub] = idx
				else:
					idx, h, r, t = line.split('\t')
					fact = [tuple(map(int, h.split(' '))), tuple(map(int, r.split(' '))),
					tuple(map(int, t.split(' ')))]
					kb[int(idx)] = fact
	return kb, sub_idx

def create_batches(N, batch_size, skip_idx = None, is_shuffle = True):
	batches = []
	shuffle_batch = np.arange(N)
	if skip_idx:
		shuffle_batch = list(set(shuffle_batch) - set(skip_idx))
	if is_shuffle:
		np.random.shuffle(shuffle_batch)
	M = int((N-1)/batch_size + 1)
	for i in range(M):
		batches += [shuffle_batch[i*batch_size: (i+1)*batch_size]]
	return batches

def create_batches2(file, batch_limit, batch_size):
	idx_num = []
	with open(file, 'r') as f:
		for line in f:
			line = line.replace('\n', '')
			idx_num += [int(line.split('\t')[1])]
	idx = sorted(range(len(idx_num)), key=lambda k: idx_num[k])[::-1]
	batches, max_batch, num, dyn_batch, batch_num = [], idx_num[idx[0]], 0, [], 0
	for i in range(len(idx)):
		batch_num += max_batch
		num += 1
		batches += [idx[i]]
		if batch_num >= batch_limit or num >= batch_size:
			dyn_batch += [batches]
			batch_num, num = 0, 0
			max_batch = idx_num[idx[i]]
			batches = []
		elif i + 1 == len(idx):
			dyn_batch += [batches]
	np.random.shuffle(dyn_batch)
	return dyn_batch

def array2tuple(a):
	b = ()
	for _, k in enumerate(a):
		if k == 0:
			break
		else:
			b += (k, )
	return b

def idx2word(a, dic2):
	return ' '.join([dic2[w] for w in a])

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = ''
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    longest = c
                    lcs_set = S[i-c+1:i+1]
                elif c == longest:
                    lcs_set = S[i-c+1:i+1]
    return lcs_set

def obtain_story(que, kb, sub_idx, dic = None, is_que = True, with_sub = True,
	entity_pos = False, alias = None, is_train = True, sub_ngram=None):
	def is_sublist(a, b):
		if a in b:
			return True
	def squeeze_story(a, direction, with_sub):
		if direction == 1:
			if with_sub:
				story = a[0] + a[1]
			else:
				story = a[1]
			obj = a[2]
		elif direction == -1:
			if with_sub:
				story = a[2] + a[1] + (1, )
			else:
				story = a[1] + (1, )
			obj = a[0]
		return story, obj
	def squeeze_story_WebQA(kb, sub_idx, c, dic2, idx, with_sub):
		a = kb[c]
		if with_sub:
			if idx == 0:
				stories, mids, objs, mid = [], [], [], ()
				story = a[0] + a[1]
				obj = a[2]
				obj_text = idx2word(a[2], dic2)
				if re.search('^m\.|^g\.', obj_text):
					#print(idx2word(story, dic2))
					#print(obj_text)
					if a[2] in sub_idx:
						for c in sub_idx[a[2]]:
							#print(c)
							s, o = squeeze_story_WebQA(kb, sub_idx, c, dic2, idx+1, with_sub)
							story_s = a[0] + s
							mid = a[2]
							obj = o
							#print(idx2word(s, dic2))
							#print(idx2word(o, dic2))
							stories += [story_s]
							mids += [mid]
							objs += [obj]
					#print('\n')
				else:
					stories = [story]
					mids = [()]
					objs = [obj]
			elif idx == 1:
				story = a[1]
				obj = a[2]
				return story, obj

			return stories, mids, objs
		else:
			if re.search('^m\.|^g\.', idx2word(a[0], dic2)):
				stories = a[1] + a[2]
				objs = None
			elif re.search('type', idx2word(a[1], dic2)):
				stories = a[1] + a[2]
				objs = None
			else:
				stories = a[1]
				objs = a[2]
			return [stories], [()], [objs]

	dic, dic2 = dic
	candidate, prev_obj = [], []
	if not sub_ngram:
		threshold = 1
	elif isinstance(sub_ngram, list):
		threshold = 0
	else:
		threshold = 5
	limit = 10000 if is_train else 1e10
	if is_que:
		if alias:
			#print(entity_pos)
			for entity in entity_pos:
				# print(entity)
				entity_idx = tuple([dic[w] for w in entity.split(' ') if w in dic])
				#print(entity_idx)
				if entity_idx in sub_idx:
					candidate += sub_idx[entity_idx]
				if (entity_idx not in sub_idx) and (entity in alias):
					entity_alias = list(alias[entity])
					# print(entity_alias)
					if len(entity_alias) > 3:
						ent_len = [len(lcs(entity, alia))*1./len(alia) for
							alia in entity_alias]
						entity_alias = [entity_alias[i] for i in
							sorted(range(len(ent_len)), key=lambda k: ent_len[k])[::-1][:3]]
						# print(entity_alias)
					for entity in entity_alias:
						#print(entity)
						entity_idx = tuple([dic[w] for w in entity.split(' ') if w in dic])
						if entity_idx in sub_idx:
							candidate += sub_idx[entity_idx]
			if len(candidate) == 0:
				candidate = np.random.randint(len(kb), size = 10)
			elif len(candidate) > limit:
				candidate = random.sample(candidate, limit)
		elif threshold == 1:
			for i in range(len(que)):
				for j in range(i+1, len(que)+1):
					if que[i:j] in sub_idx:
						candidate += sub_idx[que[i:j]]
		elif threshold == 5:
			sub_dis = defaultdict(int)
			time1 = time.time()
			subs = set()
			for i in range(len(que)):
				for j in range(i+1, np.min([i+5, len(que)+1])):
					if not set(que[i:j]) < set(stop_words):
						#print(idx2word(que[i:j], dic2))
						que_str = idx2word(que[i:j], dic2)
						que_ngrams = ngrams(que_str, 3)
						#print(que_ngrams)
						#print(sub_ngram.keys())
						for que_ngram in que_ngrams:
							subs = subs.union(sub_ngram[que_ngram])
						#print(subs)
						for sub in subs:
							#print(sub)
							sub_str = idx2word(sub, dic2)
							dis = editdistance.eval(sub_str, que_str)
							dis = 1. - dis*1./np.max([len(sub_str), len(que_str)])
							if sub_dis[sub] < dis:
								sub_dis[sub] = dis
			print('obtain candidate ... %s' %(time.time() - time1))
			time1 = time.time()
			subs = sorted(sub_dis.items(), key=lambda v: v[1],
				reverse=True)[:threshold]
			print('sort candidate ... %s' %(time.time() - time1))
			for sub, _ in subs:
				candidate += sub_idx[sub]
		else:
			for i in range(len(sub_ngram)):
				# print(idx2word(sub_ngram[i], dic2))
				candidate += sub_idx[sub_ngram[i]]
	else:
		time1 = time.time()
		for i in range(len(que)):
			if isinstance(que[i][0], int):
				candidate += sub_idx[que[i]]
			else:
				if que[i][0] in sub_idx:
					candidate += sub_idx[que[i][0]]
					prev_obj += [que[i][0]]*len(sub_idx[que[i][0]])
				if que[i][1] in sub_idx:
					candidate += sub_idx[que[i][1]]
					prev_obj += [que[i][0]]*len(sub_idx[que[i][1]])
				#print('!!! %s' %str(candidate))
		if not isinstance(que[i][0], int):
			if len(candidate) > limit:
				candidate, prev_obj = zip(*random.sample(list(zip(candidate,
					prev_obj)), limit))
		#print('collect candidates num %s time %s' %(time.time()-time1, len(candidate)))

	story, story2idx, idx = [], {}, 0
	mid, obj = defaultdict(list), defaultdict(list)
	#print('%s\t%s' %(len(candidate), set(candidate)))
	if not alias:
		for dc in candidate:
			direction, c = dc
			s, o = squeeze_story(kb[c], direction, with_sub)
			if s not in story2idx:
				#print(idx2word(s, dic2))
				story += [s]
				story2idx[s] = idx
				idx += 1
			obj[story2idx[s]] += [o]
		obj = [list(set(obj[i])) for i in obj]
		#exit()
		return story, None, obj
	else:
		time1 = time.time()
		for c_idx, c in enumerate(candidate):
			ss, ms, os = squeeze_story_WebQA(kb, sub_idx, c, dic2, 0, with_sub)
			for s_idx, s in enumerate(ss):
				#print('os ... %s' %str(os[s_idx]))
				if os[s_idx] is None:
					#print('yeah ~ %s' %str(prev_obj[c_idx]))
					os[s_idx] = prev_obj[c_idx]
				obj_text = idx2word(os[s_idx], dic2)
				if not re.search('^m\.|g\.', obj_text):
					if s not in story2idx:
						story += [s]
						# print(s)
						# print(idx2word(s, dic2))
						story2idx[s] = idx
						idx += 1
					if os[s_idx] not in obj[story2idx[s]]:
						obj[story2idx[s]] += [os[s_idx]]
						mid[story2idx[s]] += [ms[s_idx]]
		#print('squeeze %s' %(time.time() - time1))
				# if c_idx == 100:
				# 	stop
		obj = [obj[i] for i in obj]
		mid = [mid[i] for i in mid]
		return story, mid, obj

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

def unique_story(story, obj):
	seen = set()
	new_story, new_obj = [], defaultdict(list)
	for i in range(len(story)):
		if story[i] not in seen:
			new_story = story[i]
			seen.add(story[i])
		new_obj[i] += [obj]
	return new_story, new_obj

def obtain_xys(data, kb, sub_idx, batches, dic, dep = None, alias = None,
	is_train= True, sub_ngram = None):
	def remove_topic(obj, bx):
		return [w for w in obj if w not in bx]

	dic, dic2 = dic
	que, ans, entity_pos = data
	x = np.zeros((len(batches), 30), dtype = np.int32)#15
	e = np.zeros((len(batches), 100), dtype = np.int32)
	y = np.zeros((len(batches), 10000))
	s = np.zeros((len(batches), 10000, 100), dtype = np.int32)
	if dep and isinstance(dep, tuple):
		d = np.zeros((len(batches), 100, 10), dtype = np.int32)
		d2 = np.zeros((len(batches), 100), dtype = np.int32)
	elif dep and isinstance(dep[0][0], int):
		d = np.zeros((len(batches), 100), dtype = np.int32)
	elif dep:
		d = np.zeros((len(batches), 100, 10), dtype = np.int32)
	o, a, m = [], [], []

	max_cand, max_xlen, max_slen, max_dlen = 1, 1, 1, 1
	for i, b in enumerate(batches):
		x[i, :len(que[b])] = que[b]
		if dep and isinstance(dep, tuple):
			for k, one_d in enumerate(dep[0][b]):
				d[i, k, :len(one_d)] = one_d
				if max_dlen < len(one_d):
					max_dlen = len(one_d)
			d2[i, :len(dep[1][b])] = dep[1][b]
		elif dep and isinstance(dep[0][0], int):
			d[i, :len(dep[b])] = dep[b]
		elif dep:
			for k, one_d in enumerate(dep[b]):
				d[i, k, :len(one_d)] = one_d
				if max_dlen < len(one_d):
					max_dlen = len(one_d)
		print(idx2word(que[b], dic2))
		print('*****************')
		if max_xlen < len(que[b]):
			max_xlen = len(que[b])
		# start = entity_pos[b][0]
		# end = start +1 if len(entity_pos[b]) == 1 else entity_pos[b][1] + 1
		# e[i, start:end] = 1

		story, mid, obj = obtain_story(que[b], kb, sub_idx, [dic, dic2],
			entity_pos = entity_pos[b], alias = alias, is_train=is_train,
			sub_ngram=None)#
		#story, obj = unique_story(story, obj)
		#print('>>> answer %s '%str(ans[b]))
		for j in range(len(story)):
			slen = np.min([100, len(story[j])])
			s[i, j, :slen] = story[j][:slen]
			if max_slen < len(story[j]):
				max_slen = len(story[j])
			obj[j] = remove_topic(obj[j], que[b])
			temp = naive_get_F1([idx2word(w, dic2) for w in obj[j]], ans[b], alias=alias)
			#if temp > .6:
			y[i, j] = temp
			#print('>>>   %s\t%s' %(' '.join([idx2word(w, dic2) for w in obj[j]]), temp))
			print(idx2word(story[j], dic2))
		o += [obj]
		m += [mid]
		a += [ans[b]]

		if max_cand < len(story):
			max_cand = len(story)
		#y[i, :] = y[i, :]/np.max([np.sum(y[i, :]), 1.e-10])

	x = x[:, :max_xlen]
	e = e[:, :max_xlen]
	y = y[:, :max_cand]
	s = s[:, :max_cand, :max_slen]
	if dep and isinstance(dep, tuple):
		d2 = d2[:, :max_xlen]
		d = d[:, :max_xlen, :max_dlen]
		d = (d, d2)
	elif dep and isinstance(dep[0][0], int):
		d = d[:, :max_xlen]
	elif dep:
		d = d[:, :max_xlen, :max_dlen]
	d = d if dep else None
	if not alias:
		m = None
	return x, y, s, o, m, e, a, d

def get_F1(probs, bs, bt, bo, dic2, batch, alias = None):
	acces = []
	preds = []

	if isinstance(bt[0], tuple):
		for i in range(probs.shape[0]):
			ans = []
			rel = []
			top_index = argmax_all(probs[i, :])
			for j in top_index:
				an = bo[i][j] if j < len(bo[i]) else []
				ans += [idx2word(w, dic2) for w in an]
				rel += [idx2word(array2tuple(bs[i, j]), dic2)]

			y_out = ['***'] + list(set(rel)) + ['***'] + ['/'.join(list(set(ans)))]
			preds += [y_out]

			#print(ans)
			if alias is None:
				if set(ans) == set(bt[batch[i]]):
					acces += [1.]
				else:
					acces += [0.]
			else:
				correct = 0
				ans_alias = obtain_alias(bt[batch[i]], alias)
				for an in ans:
					# print('ans %s' %str(an))
					# print('alias %s' %str(obtain_alias([an], alias)))
					if len(obtain_alias([an], alias) & ans_alias) > 0:
						correct += 1
				precision = correct*1./np.max([len(set(ans)), 1e-10])
				recall = correct*1./len(set(bt[batch[i]]))
				f1 = np.min([1., 2*precision*recall/np.max([precision+recall, 1e-10])])
				acces += [f1]

	elif isinstance(bt[0], str):
		for i in range(probs.shape[0]):
			ans = []
			rel = []
			top_index = argmax_all(probs[i, :])
			#print(bs)
			for j in top_index:
				an = bo[i][j] if j < len(bo[i]) else []
				ans += [idx2word(w, dic2) for w in an]
				rel += [idx2word(array2tuple(bs[i, j]), dic2)]

			y_out = ['***'] + list(set(rel)) + ['***'] + ['/'.join(list(set(ans)))]
			preds += [y_out]

			rel_len = 0
			# for q in qtype:
			# 	rel_len += len(re.findall(q, rel[0]))
			#
			# pred_rel_len = 0
			# for q in qtype:
			# 	pred_rel_len += len(re.findall(q, bt[batch[i]]))
			#
			# if bt[batch[i]] in rel[0] and rel_len == pred_rel_len:
			# 	acces += [1.]
			# else:
			# 	acces += [0.]
			my_pred = re.sub('parents -1', 'children', y_out[1])
			my_pred = re.sub('children -1', 'parents', my_pred)
			my_pred = re.sub('spouse -1', 'spouse', my_pred)
			#my_pred = re.sub('-1', 'inverse', my_pred)
			#print('yeah')
			#print(my_pred)
			#print(bt[batch[i]])
			if my_pred == bt[batch[i]]:
				acces += [1.]
			else:
				acces += [0.]

	return acces, preds

def argmax_all(l, top_num = 1):
	m = sorted(l)[::-1][:top_num]
	return [i for i,j in enumerate(l) if j in m][:top_num]
	#return np.argsort(l)[::-1][:top_num]

def obtain_alias(target, alias):
    out = set()
    for t in target:
        if t in alias:
            out.update(alias[t])
        out.add(t)
    return out

def naive_get_F1(preds, ans, alias = None):
	if alias:
		correct = 0
		ans_alias = obtain_alias(ans, alias)
		for pred in preds:
			if len(obtain_alias([pred], alias) & ans_alias) > 0:
				correct += 1
		precision = correct*1./np.max([len(set(preds)), 1e-10])
		recall = correct*1./len(set(ans))
		f1 = np.min([1., 2*precision*recall/np.max([precision+recall, 1e-10])])
	else:
		precision = len(set(preds)&set(ans))*1./np.max([len(set(preds)), 1e-10])
		recall = len(set(preds)&set(ans))*1./len(set(ans))
		f1 = 2*precision*recall/np.max([precision+recall, 1e-10])
	return f1

def obtain_next_xyz(probs, bx, bs, bo, bm, ans, kb, sub_idx, dic, is_prev_log=False,
	top=3, alias = None):
	def remove_sub(obj, bx):
		return [w for w in obj if len(set(w)-set(bx)) != 0]

	dic, dic2 = dic
	y = np.zeros((len(bo), 10000))
	s = np.zeros((len(bo), 10000, 300), dtype = np.int32)
	ytable, m = [], None

	max_cand, max_slen = 1, 1
	if not is_prev_log:
		o = []
		for i in range(probs.shape[0]):
			#print(ans[i])
			idx = 0
			objs = []
			top_index = argmax_all(probs[i, :], top)
			for j in top_index:
				an = bo[i][j] if j < len(bo[i]) else []
				story, obj = obtain_story(an, kb, sub_idx, is_que = False,
					with_sub = False, alias = alias)

				for k in range(len(story)):
					obj[k] = remove_sub(obj[k], bx[i])
					story_unit = array2tuple(bs[i, j, :]) + story[k]
					s[i, idx, :len(story_unit)] = story_unit
					if max_slen < len(story_unit):
						max_slen = len(story_unit)
					y[i, idx] = naive_get_F1([idx2word(w, dic2) for w in obj[k]], ans[i])
					#print('>>>  %s\t%s' %(idx2word(story_unit, dic2), y[i, idx]))
					ytable += [(i, (idx, j))]
					idx += 1
				objs += obj
			if max_cand < idx:
				max_cand = idx

			o += [objs]
			y[i, :] = y[i, :]/np.max([np.sum(y[i, :]), 1.e-10])
		y = y[:, :max_cand]
		s = s[:, :max_cand, :max_slen]
		return y, s, o, ytable
	else:
		prev_logits = np.zeros_like(probs)
		full_s = np.zeros((len(bo), 10000, 300), dtype = np.int32)
		objs, prev_s = {}, {}
		max_j_idx, max_full_slen = 1, 1
		for i in range(probs.shape[0]):
			idx = 0
			seen = {}
			top_index = argmax_all(probs[i, :], top)
			for j_idx, j in enumerate(top_index):
				prev_logits[i, j] = 1
				prev_s[(i, j_idx)] = array2tuple(bs[i, j])
				an = bo[i][j] if j < len(bo[i]) else []
				if bm:
					mid = bm[i][j] if j < len(bm[i]) else []
					an = [(an[k], mid[k]) for k in range(len(an))]
					#print(an)

				#time1 = time.time()
				if len(an) > 0:
					time1 = time.time()
					story, m, obj = obtain_story(an, kb, sub_idx, [dic, dic2],
						is_que=False, with_sub = False, alias = alias)
					#print('an num %s' %len(an))
					#print('obtain next story %s' %(time.time() - time1))

					for k in range(len(story)):
						#obj[k] = remove_sub(obj[k], bx[i])
						if story[k] not in seen:
							seen[story[k]] = idx
							idx += 1
						s[i, seen[story[k]], :len(story[k])] = story[k]
						if max_slen < len(story[k]):
							max_slen = len(story[k])
						objs[(i, j_idx, seen[story[k]])] = obj[k]
						ytable += [(i, (seen[story[k]], j))]
				#print('%s: \t%s' %(i, time.time() - time1))
			if max_cand < idx:
				max_cand = idx
			if max_j_idx < j_idx + 1:
				max_j_idx = j_idx + 1
		o = [[[]]*(max_cand*max_j_idx) for _ in range(probs.shape[0])]
		for i, j_idx, idx in objs:
			#print('>>> answer %s '%str(ans[i]))
			temp = naive_get_F1([idx2word(w, dic2)
							for w in objs[(i, j_idx, idx)]], ans[i], alias = alias)
			# if temp > .6:
			y[i, j_idx*max_cand + idx] = temp
			# print('>>>   %s\t%s\t%s' %(' '.join([idx2word(w, dic2) for w in objs[(i, j_idx, idx)][:3]]), ans[i], temp))
			o[i][j_idx*max_cand + idx] = objs[(i, j_idx, idx)]
			story_unit = prev_s[(i, j_idx)] + array2tuple(s[i, idx])
			full_s[i, j_idx*max_cand + idx, :len(story_unit)] = story_unit
			if max_full_slen < len(story_unit):
				max_full_slen = len(story_unit)

		y = y[:, :max_cand*max_j_idx]
		full_s = full_s[:, :max_cand*max_j_idx, :max_full_slen]

		s = s[:, :max_cand, :max_slen]
		# for k in range(full_s.shape[1]):
		# 	print('%s\t%s\t%s' %(idx2word(full_s[0, k], dic2),
		# 		[idx2word(w, dic2) for w in o[0][k][:3]], y[0, k]))
		return y, s, o, m, ytable, prev_logits, full_s

def obtain_top_xyz(preds, bs, by, bo, top_num):
	s, y, o = [], [], []
	for i in range(preds.shape[0]):
		top_idx = np.argsort(preds[i, :])[::-1][:top_num]
		s += [bs[i, top_idx, :]]
		y += [by[i, top_idx]]
		o += [[bo[i][j] for j in range(len(bo[i])) if j in top_idx]]
	s = np.stack(s, axis = 0)
	y = np.stack(y, axis = 0)
	return s, y, o

def obtain_prev_y(prev_y, ytable, y):
	for i, j in ytable:
		y[i, j[1]] += prev_y[i, j[0]]
	y /= np.expand_dims(np.maximum(np.sum(y, 1), 1.e-10), 1)
	return y

def initialize_vocab(dic, path):
	vocab = np.random.uniform(-0.1, 0.1, (len(dic), 300))
	seen = 0

	gloves = zipfile.ZipFile(path)
	for glove in gloves.infolist():
		with gloves.open(glove) as f:
			for line in f:
				if line != "":
					splitline = line.split()
					word = splitline[0].decode('utf-8')
					embedding = splitline[1:]
					if word in dic and len(embedding) == 300:
						temp = np.array([float(val) for val in embedding])
						vocab[dic[word], :] = temp/np.sqrt(np.sum(temp**2))
						seen += 1

	vocab = vocab.astype(np.float32)
	vocab[0, :]  = 0.
	print("pretrained vocab %s among %s" %(seen, len(dic)))
	return vocab

def print_pred(preds, shuffle_batch, evals = None):
	shuffle_batch = np.concatenate(shuffle_batch)
	idx = sorted(range(len(shuffle_batch)), key = lambda x: shuffle_batch[x])
	pred_text = []
	for i in range(len(idx)):
		if evals:
			text = []
			for j in range(len(preds[idx[i]])):
				w = preds[idx[i]][j]
				text += [w]
			pred_text += ['%s\t%s\t%s' %(shuffle_batch[idx[i]]+1, evals[idx[i]],
							'\t'.join(text))]
		else:
			pred_text += ['%s\t%s' %(shuffle_batch[idx[i]]+1, preds[idx[i]])]
	return pred_text

def save_pred(file, preds):
	with open(file, 'w') as f:
		f.write('\n'.join(preds))
	f.close()
