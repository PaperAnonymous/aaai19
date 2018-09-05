import pickle
import random
import re
import numpy as np
from collections import defaultdict

datapath = './trained_model/2018-01-06-09-37-34'
global wrong_mistake
wrong_mistake = [296, 583, 1284, 1407, 8720, 9243, 9616, 9897]
global qtype2rel
qtype2rel = {'actor_to_movie': 'starred actors -1', 'director_to_movie': 'directed by -1',
'movie_to_actor': 'starred actors', 'movie_to_director': 'directed by',
'movie_to_genre': 'has genre', 'movie_to_language': 'in language',
'movie_to_tags': 'has tags', 'movie_to_writer': 'written by',
'movie_to_year': 'release year', 'tag_to_movie': 'has tags -1',
'writer_to_movie': 'written by -1', 'movie_to_imdbrating': 'has imdb rating',
'movie_to_imdbvotes': 'has imdb votes'}
qtype = set(['starred actors', 'directed by', 'has genre', 'in language', 'has tags',
	'written by', 'release year', 'has imdb rating', 'has imdb votes'])

def convert_numtxt_to_numpkl(file, outfile):
	idx_num = {}
	with open(file, 'r') as f:
		for line in f:
			line = line.replace('\n', '')
			idx, num = line.split('\t')
			idx_num[int(idx)] = int(num)
	idx_num = [idx_num[w] for w in range(len(idx_num))]
	pickle.dump(idx_num, open(outfile, 'w'), protocol = 2)

#convert_numtxt_to_numpkl('%s/test_num.txt' %datapath, '%s/test_num.pkl' %datapath)

def generate_noise(file, outfile, dic):
	def obtain_noise_phrase(dic):
		phrase = []
		phrase_num = random.randint(0, 5)
		for _ in range(phrase_num):
			phrase += random.sample(dic, 1)
		return phrase

	g = open(outfile, 'w')
	with open(file, 'r') as f:
		for line in f:
			line = line.replace('\n', '')
			q, a = line.split('\t')
			before = ' '.join(obtain_noise_phrase(dic))
			after = ' '.join(obtain_noise_phrase(dic))
			q = before + ' ' + q + ' ' + after
			g.write('\t'.join([q, a]) + '\n')
	g.close()

def obtain_dic(file):
	dic = set()
	ent = set()
	rel = set()
	with open(file, 'r') as f:
		for line in f:
			line = line.replace('\n', '').lower()
			h, r, t = line.split('|')
			ent.add(h)
			ent.add(t)
			rel.add(r)
			rel.add(r+'-1')
			dic = dic.union(set(r.split('_')))
			#dic = dic.union(set(h.split('_')))
	print('entity %s\t relation %s' %(len(ent), len(rel)))
	return dic

def check_hit1(ans_file, pred_file, q_file=None, check_ans = True):
	answer = []
	topics = []
	if check_ans:
		with open(ans_file) as f:
			for line in f:
				a2idx = ()
				line = line.replace('\n', '').lower()
				q_txt, a_txt = line.split('\t')

				a_txt = re.sub("'s", " 's", a_txt)
				a_txt = a_txt.split('|')
				for w in a_txt:
					a2idx += (w, )
				answer += [a2idx]
	else:
		with open(q_file) as f:
			for line in f:
				a2idx = ()
				line = line.replace('\n', '').lower()
				q_txt, a_txt = line.split('\t')

				topic = re.findall('(?<=\[)[^\]]+', q_txt)[0]
				topics += [topic]
		answer = []
		with open(ans_file) as f:
			for line in f:
				line = line.replace('\n', '')
				golden = []
				line = line.split('_')
				for i in range(int(np.ceil(len(line)/3.))):
					#print(line)
					golden += [qtype2rel['_'.join(line[i*2: i*2+3])]]
				#print(golden)
				answer += [' '.join(golden)]

	acc = 0
	with open(pred_file) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')
			#print(line)
			if check_ans:
				a_txt = line.split('\t')[-1]

				pred_answer = a_txt.split('/')
				pred_answer = random.sample(pred_answer, 1)[0]
				if pred_answer in answer[line_idx]:
					#print('%s\t%s\t%s\t%s' %(line_idx+1, pred_answer, answer[line_idx], pred_answer2))
					acc += 1.
				else:
					pass#print('%s\t%s\t%s' %(line_idx+1, pred_answer, answer[line_idx]))
			else:
				#pred_answer = line.split('\t')[3]
				pred_answer = line.split('***')[1]
				pred_answer = pred_answer.strip('\t')
				#pred_answer = re.sub('what', '-1', pred_answer)
				rel_len = 0
				for q in qtype:
					rel_len += len(re.findall(q, pred_answer))

				pred_rel_len = 0
				for q in qtype:
					pred_rel_len += len(re.findall(q, answer[line_idx]))

				golden_answer = topics[line_idx] + ' '+answer[line_idx]
				golden_answer = re.sub("'s", " 's", golden_answer)
				if pred_answer == golden_answer:
					acc += 1.
					# print('%s\t%s\t%s' %(line_idx+1, pred_answer,
					# 	golden_answer))
				else:
					pass
					# print('%s\t%s\t%s' %(line_idx+1, pred_answer,
					# 	golden_answer))
			# if line_idx == 200:
			# 	break
	acc /= (line_idx+1)
	#print('hits@1 : %s' %acc)
	return acc

def question_type(ans_file, rel_file, pred_file):
	answer =[]
	topics = []
	with open(ans_file) as f:
		for line in f:
			a2idx = ()
			line = line.replace('\n', '').lower()
			q_txt, a_txt = line.split('\t')

			a_txt = re.sub("'s", " 's", a_txt)
			a_txt = a_txt.split('|')
			for w in a_txt:
				a2idx += (w, )
			answer += [a2idx]
			topic = re.findall('(?<=\[)[^\]]+', q_txt)[0]
			topics += [topic]

	rel_count = defaultdict(int)
	rel = []
	golden_rel = []
	scores = []
	rel_trans = {}
	with open(rel_file) as f:
		for line in f:
			line = line.replace('\n', '')
			rel += [len(line.split('_'))]
			rel_count[len(line.split('_'))] += 1
			line_list = line.split('_')
			golden = []
			for i in range(int(np.ceil(len(line_list)/3.))):
				#print(line)
				golden += [qtype2rel['_'.join(line_list[i*2: i*2+3])]]
			golden_rel += [' '.join(golden)]
			rel_trans[' '.join(golden)] = line

	hit1 = defaultdict(int)
	rel_acc = defaultdict(int)
	q_type = defaultdict(int)
	breakdown = defaultdict(int)
	with open(pred_file) as f:
		for line_idx, line in enumerate(f):
			tmp = False
			line = line.replace('\n', '')
			pred_rel = line.split('***')[1]
			pred_rel = pred_rel.strip('\t')
			a_txt = line.split('\t')[-1]
			pred_answer = a_txt.split('/')
			pred_answer = random.sample(pred_answer, 1)[0]
			if pred_answer in answer[line_idx]:
				q_type[golden_rel[line_idx]] += 1
				hit1[rel[line_idx]] += 1
				scores += ['1']
			else:
				q_type[golden_rel[line_idx]] += 0
				scores += ['0']
				#print('%s\t%s\t%s' %(line_idx+1, answer[line_idx], pred_answer))
				#print('%s\t%s\t%s' %(line_idx+1, golden_rel[line_idx], pred_rel))
			rel_len = 0
			for q in qtype:
				rel_len += len(re.findall(q, pred_rel))
			breakdown[(int(rel[line_idx]/2), rel_len)] += 1

			pred_rel_len = 0
			for q in qtype:
				pred_rel_len += len(re.findall(q, golden_rel[line_idx]))

			golden_answer = topics[line_idx] + ' '+golden_rel[line_idx]
			golden_answer = re.sub("'s", " 's", golden_answer)

			if pred_rel == golden_answer:
				rel_acc[rel[line_idx]] += 1
				#print('%s\t%s\t%s' %(line_idx+1, golden_rel[line_idx], pred_rel))
			# if rel_len == 0:
			# 	print(pred_rel)

	print(breakdown)
	print(hit1)
	print(np.sum([q_type[i] for i in q_type]))
	rel_out = []
	hit1_out = []
	relacc_out = []
	for i in sorted(hit1):
		rel_out += [i]
		hit1_out += [str(np.round(hit1[i]*1./rel_count[i], 3))]
		relacc_out += [str(np.round(rel_acc[i]*1./rel_count[i], 3))]
	print(' '.join(map(str, rel_out)))
	print(' '.join(hit1_out))
	print(' '.join(relacc_out))
	rels = []
	acc = []
	for j, i in enumerate(sorted(q_type.keys())):
		rels += ['%s.%s' %(j, rel_trans[i])]
		acc += [np.round(q_type[i]*1./golden_rel.count(i), 3)]
	print(len(rels))
	print(' '.join(rels))
	print(' '.join(map(str, acc)))
	return scores

def obtain_eval(ans_path):
	scores = []
	with open(ans_path) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')
			scores += [re.findall('(?<=test eval: )[\d.]+', line)[0]]

	print(' '.join(scores))

def topic_accuracy(pred_file, rel_file, ans_file):
	answer = []
	with open(rel_file) as f:
		for line in f:
			line = line.replace('\n', '')
			golden = []
			line = line.split('_')
			for i in range(int(np.ceil(len(line)/3.))):
				golden += [qtype2rel['_'.join(line[i*2: i*2+3])]]
			answer += [' '.join(golden)]

	topics = []
	with open(ans_file) as f:
		for line in f:
			a2idx = ()
			line = line.replace('\n', '').lower()
			q_txt, a_txt = line.split('\t')

			topic = re.findall('(?<=\[)[^\]]+', q_txt)[0]
			topics += [topic]

	acc = 0
	with open(pred_file) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')

			pred_answer = line.split('***')[1]
			pred_answer = pred_answer.strip('\t')

			rel_len = {}
			for q in qtype:
				q_1 = q + ' -1'
				rel_1 = -1
				if re.search(q_1, pred_answer):
					rel_1 = re.search(q_1, pred_answer).start()
					if rel_1:
						rel_len[rel_1] = q_1
				if re.search(q, pred_answer):
					rel_2 = re.search(q, pred_answer).start()
					if rel_2 and rel_2 != rel_1:
						rel_len[rel_2] = q
			rel_len = sorted(rel_len.items(), key=lambda k: k[0])
			rel_len = ' ' + ' '.join([w for _, w in rel_len])
			pred_answer = re.sub(rel_len, '', pred_answer)

			if pred_answer == topics[line_idx]:
				acc += 1

	print('topic entity accuracy ... %s' %(acc*1./(line_idx+1)))

def order_mid():
	mids = {}
	with open('data/mix-hop/vanilla/qa_test_mid.txt', 'r') as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')
			idx = int(line.split('\t')[0])
			mids[idx] = '\t'.join(line.split('\t')[1:])
	g = open('./data/mix-hop/vanilla/qa_test_mid2.txt', 'w')
	for i in range(len(mids)):
		g.write('%s\n' %mids[i])
	g.close()

def topic_entity_acc(ans_file, pred_file):
	topics = []
	with open(ans_file) as f:
		for line in f:
			a2idx = ()
			line = line.replace('\n', '').lower()
			q_txt, a_txt = line.split('\t')

			topic = re.findall('(?<=\[)[^\]]+', q_txt)[0]
			topics += [topic]
	acc = 0
	with open(pred_file) as f:
		for line_idx, line in enumerate(f):
			line = line.replace('\n', '')
			if topics[line_idx] in line.split('\t'):
				acc += 1
	print(acc*1./(line_idx+1))

def read_rel(rel_file):
	rel_count = defaultdict(int)
	rel = defaultdict(set)
	with open(rel_file) as f:
		for line in f:
			line = line.replace('\n', '')
			rel_count[len(line.split('_'))] += 1
			rel[len(line.split('_'))].add(line)
	print(rel_count)
	for r in rel:
		print('%s\t%s' %(r, len(rel[r])))

def print_acc():
	acces = []
	with open('/home/yunshi/Dropbox/Multi-hopQA/trained_model/2018-03-22-03-39-56/result_Acc_beamsearch.txt') as f:
		for line_idx, line in enumerate(f):
			line = line.strip()
			acc = re.findall('(?<=test eval: )[^ ]+', line)[0]
			acces += [acc]
	print(' '.join(acces))

#dic = obtain_dic('./data/kb.txt')
# generate_noise('./data/1-hop/vanilla/qa_test.txt',
# 	'./data/1-hop/vanilla/qa_test_noise.txt', dic)


path = './trained_model/others/pred_bilstm_dam_top1_test.txt'
print_acc()
# acces = []
# for _ in range(100):
# 	acc = check_hit1('./data/mix-hop/vanilla/qa_test.txt',
# 		path)
# 	acces += [acc]
# print(min(acces))

# acc = check_hit1('./data/mix-hop/qa_test_qtype.txt',
# 	path, './data/mix-hop/vanilla/qa_test.txt', False)
# print(acc)

# max_scores = 0.
# final_scores = []
# for _ in range(1):
# 	scores = question_type('./data/mix-hop/vanilla/qa_test.txt',
# 		'./data/mix-hop/qa_test_qtype.txt',
# 		path)
# 	if max_scores < np.mean(map(float, scores)):
# 		final_scores = scores
# 		max_scores = np.mean(map(float, scores))
# g = open(re.sub('pred', 'score', path), 'w')
# g.write('\n'.join(final_scores))
# g.close()

#order_mid()
#obtain_eval('./trained_model/2018-03-18-09-16-06/result_update_lstm_current_notdiff_top3.txt')
# topic_entity_acc('./data/mix-hop/ntm/qa_test.txt',
# 	'./data/mix-hop/audio/qa_test_mid.txt')

# topic_accuracy('./trained_model/2018-04-11-12-32-25/pred7_top1_test.txt',
# 	'./data/mix-hop/qa_test_qtype.txt', './data/mix-hop/ntm/qa_test.txt')

#read_rel('./data/mix-hop/qa_dev_qtype.txt')
