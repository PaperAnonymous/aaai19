import os
import re
import random
from scipy import stats

def eval_accuracy(golden_path, data_path):
	golden_path = os.path.join(golden_path, 'qa_test.txt')
	answers = []
	with open(golden_path) as f:
		for line_idx, line in enumerate(f):
			line = line.strip().lower()
			q, a = line.split('\t')
			a = re.sub('\_', ' ', a)
			answers += [a.split('|')]
	p = []
	with open(data_path) as f:
		for line_idx, line in enumerate(f):
			line = line.strip().lower()
			pred = line.split('\t***\t')[-1].split('/')
			#pred = random.sample(pred, 1)
			# print(pred)
			# print(answers[line_idx])

			if len(set(pred) & set(answers[line_idx])) > 0:# or float(line.split('\t***\t')[0].split('\t')[-1]) == 1:
				p += [1]
			else:
				p += [0]
	print('Accuracy %g ...' %(sum(p)*1./len(answers)))
	return p

def t_test(a, b):
	l00, l01, l10, l11 = 0, 0, 0, 0
	for i in range(len(a)):
		if a[i] == 0 and b[i] == 0:
			l00 += 1
		elif a[i] == 0 and b[i] == 1:
			l01 += 1
		elif a[i] == 1 and b[i] == 0:
			l10 += 1
		else:
			l11 += 1
	eval = (l10 - l01 - 0.5)**2/(l10 + l01)
	print('eval: %s chi-square(0.05, 1): 3.84 chi-square(0.1, 1): 2.70' %eval)

def hop_acc(data_path):
	golden_nums = []
	goldens = []
	rels = ['cause of death', 'religion', 'gender', 'profession', 'institution',
	'place of birth', 'parents', 'location', 'nationality', 'spouse', 'place of death',
	'children', 'ethnicity']
	rels = '|'.join(rels)
	hop_acc = 0
	with open('data/mix-hop/PathQuestions/qa_test_qtype.txt') as f:
		for line_idx, line in enumerate(f):
			line = line.strip()
			golden_num = len(re.findall('#', line))/2
			golden_nums += [golden_num]
			goldens += [line]
	with open(data_path) as f:
		for line_idx, line in enumerate(f):
			line = line.strip()
			line = re.findall('(?<=\t\*\*\*\t)[^\t]+', line)[0]
			pred_num = len(re.findall(rels, line))

			if golden_nums[line_idx] == pred_num:
				hop_acc += 1
			else:
				print('%s\t%s\t%s\t%s' %(line, pred_num, goldens[line_idx], golden_nums[line_idx]))
	print('hop acc %s' %hop_acc)

a = eval_accuracy('data/mix-hop/PathQuestions', 'trained_model/2018-07-30-10-49-19/pred_Prune_copy_copy_test.txt')
#b = eval_accuracy('data/mix-hop/WC2014', 'trained_model/2018-08-13-10-13-48/predsimple150_top1_test.txt')
#t_test(a, b)
#hop_acc('trained_model/2018-07-30-10-49-19/pred_Acc_test.txt')
